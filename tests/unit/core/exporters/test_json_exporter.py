"""Tests for the JSON exporter (manifest, identity model, parquet sidecar, preview)."""

from __future__ import annotations

import json
import uuid

from lazylabeltext.core.exporters.json_exporter import JSONExporter
from lazylabeltext.core.models import (
    Category,
    Chunk,
    ChunkingRun,
    ConvertedDocument,
    Label,
    Rubric,
)


def _seed_corpus(db, *, with_embedding: bool = True) -> tuple[int, int, int]:
    """Insert one doc, one chunking run, one labeled chunk; return ids."""
    doc_id = db.insert_document(
        ConvertedDocument(filename="doc.txt", format="txt", full_text="hello", status="parsed")
    )
    rid = db.insert_rubric(
        Rubric(name="R", version=1, categories=[Category(name="cat", definition="d")])
    )
    run_id = db.insert_chunking_run(
        ChunkingRun(document_id=doc_id, strategy="structural", n_chunks=1)
    )
    chunk = Chunk(
        document_id=doc_id,
        chunking_run_id=run_id,
        text="hello world",
        char_start=0,
        char_end=11,
        token_count=2,
    )
    chunk_id = db.insert_chunk(chunk)
    if with_embedding:
        db.set_chunk_embedding(chunk_id, [0.1, 0.2, 0.3], "test-model")
    db.insert_label(
        Label(
            chunk_id=chunk_id,
            rubric_version_id=rid,
            predicted_categories=["cat"],
            confidence_per_category={"cat": 0.9},
            rationale="r",
            composite_confidence=0.9,
            llm_model="test-llm",
        )
    )
    return doc_id, rid, chunk_id


class TestProjectUuid:
    def test_get_or_create_is_stable(self, sample_database):
        u1 = sample_database.get_or_create_project_uuid()
        u2 = sample_database.get_or_create_project_uuid()
        assert u1 == u2
        # Real UUID, not just a non-empty string.
        uuid.UUID(u1)

    def test_distinct_per_db(self, tmp_path):
        from lazylabeltext.core.database import Database

        a = Database(str(tmp_path / "a.db"))
        b = Database(str(tmp_path / "b.db"))
        try:
            assert a.get_or_create_project_uuid() != b.get_or_create_project_uuid()
        finally:
            a.close()
            b.close()


class TestExportIdentity:
    def test_manifest_carries_project_uuid_and_join_keys(
        self, sample_database, tmp_path
    ):
        _, rid, _ = _seed_corpus(sample_database, with_embedding=False)
        out = tmp_path / "corpus.json"

        JSONExporter(sidecar_parquet=False).export(
            sample_database, rid, str(out)
        )
        data = json.loads(out.read_text(encoding="utf-8"))

        assert data["manifest"]["project_uuid"] == sample_database.get_or_create_project_uuid()
        assert data["manifest"]["join_keys"]["primary"] == "chunk_uuid"
        assert "project_uuid" in data["manifest"]["join_keys"]["fallback"]

    def test_chunk_uuid_is_uuid5_of_project_and_chunk_id(
        self, sample_database, tmp_path
    ):
        _, rid, chunk_id = _seed_corpus(sample_database, with_embedding=False)
        out = tmp_path / "corpus.json"
        JSONExporter(sidecar_parquet=False).export(
            sample_database, rid, str(out)
        )
        data = json.loads(out.read_text(encoding="utf-8"))

        project_uuid = data["manifest"]["project_uuid"]
        expected = str(uuid.uuid5(uuid.UUID(project_uuid), str(chunk_id)))
        assert data["chunks"][0]["chunk_uuid"] == expected

    def test_chunk_uuid_stable_across_exports(self, sample_database, tmp_path):
        _, rid, _ = _seed_corpus(sample_database, with_embedding=False)

        out1 = tmp_path / "a.json"
        out2 = tmp_path / "b.json"
        ex = JSONExporter(sidecar_parquet=False)
        ex.export(sample_database, rid, str(out1))
        ex.export(sample_database, rid, str(out2))

        a = json.loads(out1.read_text(encoding="utf-8"))
        b = json.loads(out2.read_text(encoding="utf-8"))
        assert a["chunks"][0]["chunk_uuid"] == b["chunks"][0]["chunk_uuid"]
        assert a["manifest"]["project_uuid"] == b["manifest"]["project_uuid"]


class TestParquetSidecar:
    def test_parquet_has_uuid_columns(self, sample_database, tmp_path):
        pq = _try_import_parquet()
        if pq is None:
            return  # pyarrow optional in dev environments

        _, rid, _ = _seed_corpus(sample_database, with_embedding=True)
        out = tmp_path / "corpus.json"
        JSONExporter(sidecar_parquet=True).export(
            sample_database, rid, str(out)
        )

        sidecar = tmp_path / "corpus_embeddings.parquet"
        assert sidecar.exists()

        table = pq.read_table(str(sidecar))
        names = table.schema.names
        assert "chunk_uuid" in names
        assert "project_uuid" in names
        assert "chunk_id" in names
        assert "embedding" in names

        meta = table.schema.metadata or {}
        assert b"project_uuid" in meta


class TestCountChunksWithEmbeddings:
    def test_counts_only_labeled_chunks_with_embeddings(self, sample_database):
        _, rid, _ = _seed_corpus(sample_database, with_embedding=True)
        # A second labeled chunk without an embedding shouldn't bump the count.
        doc = sample_database.get_all_documents()[0]
        run_id = sample_database.insert_chunking_run(
            ChunkingRun(document_id=doc.id, strategy="structural", n_chunks=1)
        )
        cid = sample_database.insert_chunk(
            Chunk(
                document_id=doc.id,
                chunking_run_id=run_id,
                text="no embedding",
                char_start=0,
                char_end=12,
                token_count=2,
            )
        )
        sample_database.insert_label(
            Label(
                chunk_id=cid,
                rubric_version_id=rid,
                predicted_categories=["cat"],
                confidence_per_category={"cat": 0.5},
                rationale="",
                composite_confidence=0.5,
                llm_model="m",
            )
        )
        assert sample_database.count_chunks_with_embeddings(rid) == 1

    def test_zero_when_no_labels(self, sample_database):
        rid = sample_database.insert_rubric(
            Rubric(name="R", version=1, categories=[Category(name="c", definition="d")])
        )
        assert sample_database.count_chunks_with_embeddings(rid) == 0


class TestPreview:
    def test_preview_does_not_write_parquet(self, sample_database, tmp_path):
        _, rid, _ = _seed_corpus(sample_database, with_embedding=True)

        # Cheap preview: file system is untouched.
        before = set(tmp_path.iterdir())
        data = JSONExporter().build_preview(sample_database, rid, max_chunks=1)
        after = set(tmp_path.iterdir())
        assert before == after

        assert "manifest" in data and "chunks" in data
        assert len(data["chunks"]) == 1
        assert data["manifest"]["preview"]["rendered_chunks"] == 1

    def test_preview_caps_chunk_count(self, sample_database, tmp_path):
        # Seed two labeled chunks
        doc_id = sample_database.insert_document(
            ConvertedDocument(filename="d.txt", format="txt", full_text="x", status="parsed")
        )
        rid = sample_database.insert_rubric(
            Rubric(name="R", version=1, categories=[Category(name="c", definition="d")])
        )
        run_id = sample_database.insert_chunking_run(
            ChunkingRun(document_id=doc_id, strategy="structural", n_chunks=2)
        )
        for txt in ("alpha", "beta"):
            cid = sample_database.insert_chunk(
                Chunk(
                    document_id=doc_id,
                    chunking_run_id=run_id,
                    text=txt,
                    char_start=0,
                    char_end=len(txt),
                    token_count=1,
                )
            )
            sample_database.insert_label(
                Label(
                    chunk_id=cid,
                    rubric_version_id=rid,
                    predicted_categories=["c"],
                    confidence_per_category={"c": 0.8},
                    rationale="",
                    composite_confidence=0.8,
                    llm_model="m",
                )
            )

        data = JSONExporter().build_preview(sample_database, rid, max_chunks=1)
        assert len(data["chunks"]) == 1
        assert data["manifest"]["total_chunks"] == 2

    def test_preview_sidecar_block_describes_schema_when_embeddings_exist(
        self, sample_database
    ):
        _, rid, _ = _seed_corpus(sample_database, with_embedding=True)
        data = JSONExporter().build_preview(sample_database, rid, max_chunks=1)
        sidecar = data["manifest"]["embeddings"]["sidecar"]
        assert sidecar is not None
        assert sidecar["format"] == "parquet"
        assert "preview_only" in sidecar


def _try_import_parquet():
    try:
        import pyarrow.parquet as pq  # noqa: F401

        return pq
    except ImportError:
        return None
