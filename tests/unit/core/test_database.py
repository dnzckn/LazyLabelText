"""Tests for the SQLite database layer."""

from __future__ import annotations

from lazylabeltext.core.models import (
    AuditEvent,
    Category,
    Chunk,
    ChunkingRun,
    ConvertedDocument,
    Heading,
    Label,
    Rubric,
)


class TestDocuments:
    def test_insert_and_retrieve(self, sample_database):
        doc = ConvertedDocument(
            filename="test.pdf",
            format="pdf",
            full_text="Hello world",
            headings=[Heading(level=1, text="Title", char_start=0, char_end=5)],
            status="parsed",
        )
        doc_id = sample_database.insert_document(doc)
        assert doc_id > 0

        retrieved = sample_database.get_document(doc_id)
        assert retrieved is not None
        assert retrieved.filename == "test.pdf"
        assert retrieved.full_text == "Hello world"
        assert len(retrieved.headings) == 1
        assert retrieved.headings[0].level == 1

    def test_get_all_documents(self, sample_database):
        for name in ["a.txt", "b.txt", "c.txt"]:
            sample_database.insert_document(
                ConvertedDocument(filename=name, format="txt", status="parsed")
            )
        docs = sample_database.get_all_documents()
        assert len(docs) == 3

    def test_update_status(self, sample_database):
        doc_id = sample_database.insert_document(
            ConvertedDocument(filename="x.txt", format="txt", status="pending")
        )
        sample_database.update_document_status(doc_id, "parsed")
        doc = sample_database.get_document(doc_id)
        assert doc.status == "parsed"


class TestRubrics:
    def test_insert_and_retrieve(self, sample_database):
        rubric = Rubric(
            name="Test",
            version=1,
            categories=[
                Category(name="cat1", definition="d1"),
                Category(name="cat2", definition="d2"),
            ],
        )
        rid = sample_database.insert_rubric(rubric)
        assert rid > 0

        r = sample_database.get_rubric(rid)
        assert r is not None
        assert r.name == "Test"
        assert len(r.categories) == 2
        assert r.categories[0].name == "cat1"

    def test_get_latest_rubric(self, sample_database):
        sample_database.insert_rubric(
            Rubric(name="R", version=1, categories=[Category(name="a")])
        )
        sample_database.insert_rubric(
            Rubric(name="R", version=2, categories=[Category(name="b")])
        )

        latest = sample_database.get_latest_rubric()
        assert latest is not None
        assert latest.version == 2


class TestChunks:
    def test_insert_and_retrieve(self, sample_database):
        doc_id = sample_database.insert_document(
            ConvertedDocument(filename="t.txt", format="txt", status="parsed")
        )
        run = ChunkingRun(document_id=doc_id, strategy="structural")
        run_id = sample_database.insert_chunking_run(run)

        chunk = Chunk(
            document_id=doc_id,
            chunking_run_id=run_id,
            text="Some chunk text",
            char_start=0,
            char_end=15,
            section_path=["Section 1"],
            token_count=4,
        )
        chunk_id = sample_database.insert_chunk(chunk)
        assert chunk_id > 0

        retrieved = sample_database.get_chunk(chunk_id)
        assert retrieved is not None
        assert retrieved.text == "Some chunk text"
        assert retrieved.section_path == ["Section 1"]

    def test_get_chunks_for_document(self, sample_database):
        doc_id = sample_database.insert_document(
            ConvertedDocument(filename="t.txt", format="txt", status="parsed")
        )
        run_id = sample_database.insert_chunking_run(
            ChunkingRun(document_id=doc_id, strategy="structural")
        )

        for i in range(3):
            sample_database.insert_chunk(
                Chunk(
                    document_id=doc_id,
                    chunking_run_id=run_id,
                    text=f"chunk {i}",
                    char_start=i * 10,
                    char_end=i * 10 + 9,
                )
            )

        chunks = sample_database.get_chunks_for_document(doc_id)
        assert len(chunks) == 3


class TestLabels:
    def test_insert_and_retrieve(self, sample_database):
        # Create prerequisite records for FK constraints
        doc_id = sample_database.insert_document(
            ConvertedDocument(filename="t.txt", format="txt", status="parsed")
        )
        run_id = sample_database.insert_chunking_run(
            ChunkingRun(document_id=doc_id, strategy="structural")
        )
        chunk_id = sample_database.insert_chunk(
            Chunk(document_id=doc_id, chunking_run_id=run_id, text="test")
        )
        rubric_id = sample_database.insert_rubric(
            Rubric(name="R", version=1, categories=[Category(name="procedure")])
        )

        label = Label(
            chunk_id=chunk_id,
            rubric_version_id=rubric_id,
            predicted_categories=["procedure"],
            confidence_per_category={"procedure": 0.9},
            rationale="Step-by-step content",
            composite_confidence=0.9,
            llm_model="claude-sonnet",
        )
        lid = sample_database.insert_label(label)
        assert lid > 0

        labels = sample_database.get_labels_for_chunk(chunk_id)
        assert len(labels) == 1
        assert labels[0].predicted_categories == ["procedure"]


class TestAudit:
    def test_insert_and_retrieve(self, sample_database):
        event = AuditEvent(
            event_type="test_event",
            actor="test",
            payload={"key": "value"},
            related_chunk_ids=[1, 2],
        )
        eid = sample_database.insert_audit_event(event)
        assert eid > 0

        events = sample_database.get_audit_events("test_event")
        assert len(events) == 1
        assert events[0].payload == {"key": "value"}
        assert events[0].related_chunk_ids == [1, 2]
