"""JSON corpus exporter, with optional Parquet embedding sidecar."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from lazylabeltext import __version__
from lazylabeltext.core.database import Database
from lazylabeltext.core.exporters import ExportFormat, _register

logger = logging.getLogger("lazylabeltext")


class JSONExporter:
    """Exports labeled corpus as JSON, with optional embeddings sidecar.

    Two embedding-placement modes, controlled by the constructor:
      - inline_embeddings=True  → embeddings live inline in chunks[*].embedding
                                  (small corpora; bloats file)
      - sidecar_parquet=True    → embeddings written to <output>_embeddings.parquet
                                  keyed by chunk_id; chunk_id is also embedded in
                                  the JSON so a downstream consumer can join.
    Both can be on simultaneously (inline + sidecar) or both off (no embeddings).
    Default leans toward sidecar when pyarrow is available, otherwise inline.

    Identity model:
      - chunk_id: per-project AUTOINCREMENT id (fast, compact, joinable locally).
      - project_uuid: stable per-project UUID4 minted on first export.
      - chunk_uuid: uuid5(project_uuid, str(chunk_id)) — globally unique, stable,
        single-string key for cross-project merges without namespacing logic.
    """

    def __init__(
        self,
        inline_embeddings: bool = False,
        sidecar_parquet: bool = True,
    ) -> None:
        self.inline_embeddings = inline_embeddings
        self.sidecar_parquet = sidecar_parquet

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, db: Database, rubric_version_id: int, output_path: str) -> str:
        rubric, project_uuid = self._load_header(db, rubric_version_id)
        chunks_data, embedding_rows, embedding_dim, embedding_models = (
            self._collect_chunks(db, rubric_version_id, project_uuid, max_chunks=None)
        )

        sidecar_info: dict | None = None
        if self.sidecar_parquet and embedding_rows:
            sidecar_path = self._sidecar_path(output_path)
            wrote = self._write_parquet(
                sidecar_path,
                embedding_rows,
                embedding_dim or 0,
                sorted(embedding_models),
                project_uuid,
            )
            if wrote:
                sidecar_info = self._sidecar_manifest(
                    Path(sidecar_path).name,
                    len(embedding_rows),
                    embedding_dim,
                    sorted(embedding_models),
                )

        summary = db.get_labeling_summary(rubric_version_id)
        manifest = self._manifest(
            rubric, project_uuid, len(chunks_data), summary, sidecar_info
        )

        output = {
            "manifest": manifest,
            "rubric": self._rubric_payload(rubric),
            "chunks": chunks_data,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info("Exported %d chunks to %s", len(chunks_data), output_path)
        if sidecar_info:
            logger.info(
                "Wrote embeddings sidecar: %s rows=%d dim=%s",
                sidecar_info["path"],
                sidecar_info["rows"],
                sidecar_info["dim"],
            )
        return output_path

    def build_preview(
        self, db: Database, rubric_version_id: int, max_chunks: int = 1
    ) -> dict:
        """Build a preview dict — same shape as export(), capped to max_chunks.

        No file is written and no parquet sidecar is materialized. The
        manifest's sidecar block is populated with what *would* be written
        so the user can see the schema/keys without paying the cost.
        """
        rubric, project_uuid = self._load_header(db, rubric_version_id)
        chunks_data, embedding_rows, embedding_dim, embedding_models = (
            self._collect_chunks(
                db, rubric_version_id, project_uuid, max_chunks=max_chunks
            )
        )

        # Best-effort row count over the whole corpus, even though we only
        # serialized max_chunks of them — preview should report what export
        # would produce.
        full_summary = db.get_labeling_summary(rubric_version_id)
        full_chunk_count = full_summary["total_labels"]
        full_embedding_rows = self._count_embedding_rows(db, rubric_version_id)

        sidecar_info: dict | None = None
        if self.sidecar_parquet and full_embedding_rows > 0:
            sidecar_info = self._sidecar_manifest(
                f"{Path('preview').stem}_embeddings.parquet",
                full_embedding_rows,
                embedding_dim,
                sorted(embedding_models),
            )
            sidecar_info["preview_only"] = (
                "Schema preview — no file written. Click 'Export to File' "
                "to materialize."
            )

        manifest = self._manifest(
            rubric, project_uuid, full_chunk_count, full_summary, sidecar_info
        )
        manifest["preview"] = {
            "rendered_chunks": len(chunks_data),
            "total_chunks": full_chunk_count,
        }

        return {
            "manifest": manifest,
            "rubric": self._rubric_payload(rubric),
            "chunks": chunks_data,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_header(
        db: Database, rubric_version_id: int
    ) -> tuple[object, str]:
        rubric = db.get_rubric(rubric_version_id)
        if rubric is None:
            raise ValueError(f"Rubric version {rubric_version_id} not found")
        project_uuid = db.get_or_create_project_uuid()
        return rubric, project_uuid

    @staticmethod
    def _chunk_uuid(project_uuid: str, chunk_id: int) -> str:
        return str(uuid.uuid5(uuid.UUID(project_uuid), str(chunk_id)))

    def _collect_chunks(
        self,
        db: Database,
        rubric_version_id: int,
        project_uuid: str,
        max_chunks: int | None,
    ) -> tuple[list[dict], list[dict], int | None, set[str]]:
        chunks_data: list[dict] = []
        embedding_rows: list[dict] = []
        embedding_dim: int | None = None
        embedding_models: set[str] = set()

        rubric = db.get_rubric(rubric_version_id)
        rubric_version = rubric.version if rubric is not None else None

        for label in db.get_all_labels(rubric_version_id):
            if max_chunks is not None and len(chunks_data) >= max_chunks:
                break
            chunk = db.get_chunk(label.chunk_id)
            if chunk is None:
                continue
            doc = db.get_document(chunk.document_id)
            reviews = db.get_reviews_for_label(label.id or 0)
            cu = (
                self._chunk_uuid(project_uuid, chunk.id)
                if chunk.id is not None
                else None
            )

            chunk_entry: dict = {
                "chunk_id": chunk.id,
                "chunk_uuid": cu,
                "text": chunk.text,
                "source": {
                    "document": doc.filename if doc else "unknown",
                    "document_id": chunk.document_id,
                    "section_path": chunk.section_path,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                },
                "token_count": chunk.token_count,
                "chunk_type": chunk.chunk_type,
                "boundary_confidence": chunk.boundary_confidence,
                "manual_override": chunk.manual_override,
                "embedding_model": chunk.embedding_model,
                "label": {
                    "rubric_version": rubric_version,
                    "categories": label.predicted_categories,
                    "confidence": label.confidence_per_category,
                    "rationale": label.rationale,
                    "composite_confidence": label.composite_confidence,
                    "knn_agreement": label.knn_agreement,
                    "logprob_signal": label.logprob_signal,
                    "llm_model": label.llm_model,
                },
            }

            if self.inline_embeddings:
                chunk_entry["embedding"] = chunk.embedding

            if chunk.embedding is not None and chunk.id is not None:
                embedding_rows.append(
                    {
                        "chunk_id": chunk.id,
                        "chunk_uuid": cu,
                        "project_uuid": project_uuid,
                        "document_id": chunk.document_id or 0,
                        "document_filename": doc.filename if doc else "",
                        "embedding": chunk.embedding,
                        "embedding_model": chunk.embedding_model or "",
                    }
                )
                if embedding_dim is None:
                    embedding_dim = len(chunk.embedding)
                if chunk.embedding_model:
                    embedding_models.add(chunk.embedding_model)

            if reviews:
                review = reviews[-1]
                chunk_entry["label"]["human_review"] = {
                    "action": review.action,
                    "final_categories": review.final_categories,
                    "reviewer": review.reviewer,
                    "notes": review.notes,
                    "reviewed_at": review.reviewed_at,
                }

            chunks_data.append(chunk_entry)

        return chunks_data, embedding_rows, embedding_dim, embedding_models

    @staticmethod
    def _count_embedding_rows(db: Database, rubric_version_id: int) -> int:
        """Cheap count of how many labeled chunks have an embedding."""
        n = 0
        for label in db.get_all_labels(rubric_version_id):
            chunk = db.get_chunk(label.chunk_id)
            if chunk is not None and chunk.embedding is not None:
                n += 1
        return n

    def _manifest(
        self,
        rubric,
        project_uuid: str,
        total_chunks: int,
        summary: dict,
        sidecar_info: dict | None,
    ) -> dict:
        return {
            "project_uuid": project_uuid,
            "rubric_version": rubric.version,
            "rubric_name": rubric.name,
            "total_chunks": total_chunks,
            "labeled_chunks": summary["total_labels"],
            "reviewed_chunks": summary["reviewed"],
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_version": __version__,
            "embeddings": {
                "inline": self.inline_embeddings,
                "sidecar": sidecar_info,
            },
            "join_keys": {
                "primary": "chunk_uuid",
                "fallback": ["project_uuid", "chunk_id"],
                "note": (
                    "chunk_uuid is uuid5(project_uuid, chunk_id) — globally "
                    "unique, stable across exports of this project. chunk_id "
                    "alone is unique only within a single project."
                ),
            },
        }

    @staticmethod
    def _rubric_payload(rubric) -> dict:
        return {
            "name": rubric.name,
            "version": rubric.version,
            "categories": [
                {
                    "name": c.name,
                    "definition": c.definition,
                    "exemplars": c.exemplars,
                    "boundary_cases": c.boundary_cases,
                    "confidence_threshold": c.confidence_threshold,
                }
                for c in rubric.categories
            ],
        }

    @staticmethod
    def _sidecar_manifest(
        path_basename: str, rows: int, dim: int | None, models: list[str]
    ) -> dict:
        return {
            "path": path_basename,
            "rows": rows,
            "dim": dim,
            "models": models,
            "join_keys": ["chunk_uuid", "project_uuid", "chunk_id"],
            "format": "parquet",
        }

    @staticmethod
    def _sidecar_path(json_path: str) -> str:
        """corpus.json → corpus_embeddings.parquet (next to the JSON)."""
        p = Path(json_path)
        stem = p.stem
        return str(p.with_name(f"{stem}_embeddings.parquet"))

    @staticmethod
    def _write_parquet(
        path: str,
        rows: list[dict],
        dim: int,
        models: list[str],
        project_uuid: str,
    ) -> bool:
        """Write embeddings to Parquet; silently skip if pyarrow isn't installed.

        Compatibility choices:
          - snappy compression (universal — zstd breaks several JS/Electron
            parquet viewers and older Spark builds)
          - explicit float32 for the embedding column (half the bytes of
            float64 and enough precision for cosine retrieval)
          - explicit schema typing on every column so viewers don't have to
            re-infer
          - parquet format version 2.4 (broadest reader support)
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning(
                "pyarrow not installed — skipping embeddings sidecar. "
                "Install with: pip install pyarrow"
            )
            return False

        try:
            schema = pa.schema(
                [
                    pa.field("chunk_uuid", pa.string()),
                    pa.field("project_uuid", pa.string()),
                    pa.field("chunk_id", pa.int64()),
                    pa.field("document_id", pa.int64()),
                    pa.field("document_filename", pa.string()),
                    pa.field("embedding", pa.list_(pa.float32())),
                    pa.field("embedding_model", pa.string()),
                ]
            )
            embedding_col = [
                [float(x) for x in r["embedding"]] for r in rows
            ]
            table = pa.table(
                {
                    "chunk_uuid": pa.array(
                        [str(r["chunk_uuid"]) for r in rows], type=pa.string()
                    ),
                    "project_uuid": pa.array(
                        [str(r["project_uuid"]) for r in rows], type=pa.string()
                    ),
                    "chunk_id": pa.array(
                        [int(r["chunk_id"]) for r in rows], type=pa.int64()
                    ),
                    "document_id": pa.array(
                        [int(r["document_id"]) for r in rows], type=pa.int64()
                    ),
                    "document_filename": pa.array(
                        [str(r["document_filename"]) for r in rows],
                        type=pa.string(),
                    ),
                    "embedding": pa.array(
                        embedding_col, type=pa.list_(pa.float32())
                    ),
                    "embedding_model": pa.array(
                        [str(r["embedding_model"]) for r in rows],
                        type=pa.string(),
                    ),
                },
                schema=schema,
            )
            meta = {
                b"embedding_dim": str(dim).encode(),
                b"embedding_models": ",".join(models).encode(),
                b"project_uuid": project_uuid.encode(),
            }
            table = table.replace_schema_metadata(meta)
            pq.write_table(
                table,
                path,
                compression="snappy",
                version="2.4",
                use_dictionary=[
                    "project_uuid",
                    "document_filename",
                    "embedding_model",
                ],
            )
            return True
        except Exception as e:
            logger.warning("Failed to write Parquet sidecar: %s", e)
            return False


_register(ExportFormat.JSON, JSONExporter())
