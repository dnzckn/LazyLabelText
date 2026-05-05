"""JSON corpus exporter, with optional Parquet embedding sidecar."""

from __future__ import annotations

import json
import logging
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
    """

    def __init__(
        self,
        inline_embeddings: bool = False,
        sidecar_parquet: bool = True,
    ) -> None:
        self.inline_embeddings = inline_embeddings
        self.sidecar_parquet = sidecar_parquet

    def export(self, db: Database, rubric_version_id: int, output_path: str) -> str:
        rubric = db.get_rubric(rubric_version_id)
        if rubric is None:
            raise ValueError(f"Rubric version {rubric_version_id} not found")

        all_labels = db.get_all_labels(rubric_version_id)
        chunks_data: list[dict] = []
        embedding_rows: list[dict] = []  # chunk_id, embedding, embedding_model
        embedding_dim: int | None = None
        embedding_model_seen: set[str] = set()

        for label in all_labels:
            chunk = db.get_chunk(label.chunk_id)
            if chunk is None:
                continue

            doc = db.get_document(chunk.document_id)
            reviews = db.get_reviews_for_label(label.id or 0)

            chunk_entry = {
                "chunk_id": chunk.id,
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
                    "rubric_version": rubric.version,
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
                        "document_id": chunk.document_id or 0,
                        "document_filename": doc.filename if doc else "",
                        "embedding": chunk.embedding,
                        "embedding_model": chunk.embedding_model or "",
                    }
                )
                if embedding_dim is None:
                    embedding_dim = len(chunk.embedding)
                if chunk.embedding_model:
                    embedding_model_seen.add(chunk.embedding_model)

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

        summary = db.get_labeling_summary(rubric_version_id)

        # Optional sidecar: write embeddings to Parquet keyed by chunk_id.
        sidecar_info: dict | None = None
        if self.sidecar_parquet and embedding_rows:
            sidecar_path = self._sidecar_path(output_path)
            wrote = self._write_parquet(
                sidecar_path,
                embedding_rows,
                embedding_dim or 0,
                sorted(embedding_model_seen),
            )
            if wrote:
                sidecar_info = {
                    "path": Path(sidecar_path).name,
                    "rows": len(embedding_rows),
                    "dim": embedding_dim,
                    "models": sorted(embedding_model_seen),
                    "join_keys": ["chunk_id", "document_id"],
                    "join_key_note": (
                        "chunk_id is unique within this export (AUTOINCREMENT "
                        "primary key). When merging exports across projects, "
                        "join on (document_filename, chunk_id) or prefix "
                        "chunk_id with a project namespace."
                    ),
                    "format": "parquet",
                }

        manifest: dict = {
            "rubric_version": rubric.version,
            "rubric_name": rubric.name,
            "total_chunks": len(chunks_data),
            "labeled_chunks": summary["total_labels"],
            "reviewed_chunks": summary["reviewed"],
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_version": __version__,
            "embeddings": {
                "inline": self.inline_embeddings,
                "sidecar": sidecar_info,
            },
        }

        output = {
            "manifest": manifest,
            "rubric": {
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
            },
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

    @staticmethod
    def _sidecar_path(json_path: str) -> str:
        """corpus.json → corpus_embeddings.parquet (next to the JSON)."""
        p = Path(json_path)
        stem = p.stem
        return str(p.with_name(f"{stem}_embeddings.parquet"))

    @staticmethod
    def _write_parquet(
        path: str, rows: list[dict], dim: int, models: list[str]
    ) -> bool:
        """Write embeddings to Parquet; silently skip if pyarrow isn't installed."""
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
            table = pa.table(
                {
                    "chunk_id": [r["chunk_id"] for r in rows],
                    "document_id": [r["document_id"] for r in rows],
                    "document_filename": [r["document_filename"] for r in rows],
                    "embedding": [r["embedding"] for r in rows],
                    "embedding_model": [r["embedding_model"] for r in rows],
                }
            )
            # Stamp dim + model list as schema-level metadata so consumers
            # don't have to peek at row 0 to find them.
            meta = {
                b"embedding_dim": str(dim).encode(),
                b"embedding_models": ",".join(models).encode(),
            }
            table = table.replace_schema_metadata(meta)
            pq.write_table(table, path, compression="zstd")
            return True
        except Exception as e:
            logger.warning("Failed to write Parquet sidecar: %s", e)
            return False


_register(ExportFormat.JSON, JSONExporter())
