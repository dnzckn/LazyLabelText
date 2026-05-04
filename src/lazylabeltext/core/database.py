"""SQLite persistence layer for LazyLabelText projects."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from lazylabeltext.core.models import (
    AuditEvent,
    Category,
    Chunk,
    ChunkingRun,
    ConvertedDocument,
    Heading,
    HumanReview,
    Label,
    PageBoundary,
    Rubric,
    SectionBoundary,
)


class Database:
    """SQLite database for a LazyLabelText project."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                format TEXT NOT NULL,
                full_text TEXT,
                headings_json TEXT DEFAULT '[]',
                pages_json TEXT DEFAULT '[]',
                sections_json TEXT DEFAULT '[]',
                metadata_json TEXT DEFAULT '{}',
                status TEXT DEFAULT 'pending',
                warnings_json TEXT DEFAULT '[]',
                ingested_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS rubric_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                categories_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS chunking_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER REFERENCES documents(id),
                strategy TEXT NOT NULL,
                params_json TEXT DEFAULT '{}',
                started_at TEXT DEFAULT (datetime('now')),
                n_chunks INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER REFERENCES documents(id),
                chunking_run_id INTEGER REFERENCES chunking_runs(id),
                text TEXT NOT NULL,
                char_start INTEGER,
                char_end INTEGER,
                section_path_json TEXT DEFAULT '[]',
                token_count INTEGER,
                chunk_type TEXT,
                boundary_confidence REAL,
                manual_override_json TEXT
            );

            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER REFERENCES chunks(id),
                rubric_version_id INTEGER REFERENCES rubric_versions(id),
                predicted_categories_json TEXT DEFAULT '[]',
                confidence_json TEXT DEFAULT '{}',
                rationale TEXT,
                knn_agreement REAL,
                composite_confidence REAL,
                llm_model TEXT,
                llm_run_id TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS human_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label_id INTEGER REFERENCES labels(id),
                reviewer TEXT DEFAULT 'default',
                action TEXT NOT NULL,
                final_categories_json TEXT DEFAULT '[]',
                notes TEXT DEFAULT '',
                reviewed_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                actor TEXT DEFAULT 'system',
                timestamp TEXT DEFAULT (datetime('now')),
                payload_json TEXT DEFAULT '{}',
                related_chunk_ids_json TEXT DEFAULT '[]'
            );
        """)
        self.conn.commit()

    # --- Documents ---

    def insert_document(self, doc: ConvertedDocument) -> int:
        cur = self.conn.execute(
            """INSERT INTO documents
               (filename, format, full_text, headings_json, pages_json,
                sections_json, metadata_json, status, warnings_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc.filename,
                doc.format,
                doc.full_text,
                json.dumps([vars(h) for h in doc.headings]),
                json.dumps([vars(p) for p in doc.pages]),
                json.dumps([vars(s) for s in doc.sections]),
                json.dumps(doc.metadata),
                doc.status,
                json.dumps(doc.warnings),
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_document(self, doc_id: int) -> ConvertedDocument | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_document(row)

    def get_all_documents(self) -> list[ConvertedDocument]:
        rows = self.conn.execute(
            "SELECT * FROM documents ORDER BY ingested_at"
        ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def update_document_status(self, doc_id: int, status: str) -> None:
        self.conn.execute(
            "UPDATE documents SET status = ? WHERE id = ?", (status, doc_id)
        )
        self.conn.commit()

    def _row_to_document(self, row: sqlite3.Row) -> ConvertedDocument:
        return ConvertedDocument(
            id=row["id"],
            filename=row["filename"],
            format=row["format"],
            full_text=row["full_text"] or "",
            headings=[Heading(**h) for h in json.loads(row["headings_json"])],
            pages=[PageBoundary(**p) for p in json.loads(row["pages_json"])],
            sections=[SectionBoundary(**s) for s in json.loads(row["sections_json"])],
            metadata=json.loads(row["metadata_json"]),
            status=row["status"],
            warnings=json.loads(row["warnings_json"]),
            ingested_at=row["ingested_at"] or "",
        )

    # --- Rubrics ---

    def insert_rubric(self, rubric: Rubric) -> int:
        cur = self.conn.execute(
            """INSERT INTO rubric_versions (name, version, categories_json)
               VALUES (?, ?, ?)""",
            (
                rubric.name,
                rubric.version,
                json.dumps([vars(c) for c in rubric.categories]),
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_rubric(self, rubric_id: int) -> Rubric | None:
        row = self.conn.execute(
            "SELECT * FROM rubric_versions WHERE id = ?", (rubric_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_rubric(row)

    def get_latest_rubric(self) -> Rubric | None:
        row = self.conn.execute(
            "SELECT * FROM rubric_versions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return self._row_to_rubric(row)

    def get_all_rubric_versions(self) -> list[Rubric]:
        rows = self.conn.execute(
            "SELECT * FROM rubric_versions ORDER BY version"
        ).fetchall()
        return [self._row_to_rubric(r) for r in rows]

    def _row_to_rubric(self, row: sqlite3.Row) -> Rubric:
        return Rubric(
            id=row["id"],
            name=row["name"],
            version=row["version"],
            categories=[Category(**c) for c in json.loads(row["categories_json"])],
            created_at=row["created_at"] or "",
        )

    # --- Chunking runs ---

    def insert_chunking_run(self, run: ChunkingRun) -> int:
        cur = self.conn.execute(
            """INSERT INTO chunking_runs (document_id, strategy, params_json, n_chunks)
               VALUES (?, ?, ?, ?)""",
            (run.document_id, run.strategy, json.dumps(run.params), run.n_chunks),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def update_chunking_run_count(self, run_id: int, n_chunks: int) -> None:
        self.conn.execute(
            "UPDATE chunking_runs SET n_chunks = ? WHERE id = ?", (n_chunks, run_id)
        )
        self.conn.commit()

    def get_chunking_runs(self, document_id: int) -> list[ChunkingRun]:
        rows = self.conn.execute(
            "SELECT * FROM chunking_runs WHERE document_id = ? ORDER BY started_at",
            (document_id,),
        ).fetchall()
        return [self._row_to_chunking_run(r) for r in rows]

    def _row_to_chunking_run(self, row: sqlite3.Row) -> ChunkingRun:
        return ChunkingRun(
            id=row["id"],
            document_id=row["document_id"],
            strategy=row["strategy"],
            params=json.loads(row["params_json"]),
            started_at=row["started_at"] or "",
            n_chunks=row["n_chunks"],
        )

    # --- Chunks ---

    def insert_chunk(self, chunk: Chunk) -> int:
        cur = self.conn.execute(
            """INSERT INTO chunks
               (document_id, chunking_run_id, text, char_start, char_end,
                section_path_json, token_count, chunk_type, boundary_confidence,
                manual_override_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk.document_id,
                chunk.chunking_run_id,
                chunk.text,
                chunk.char_start,
                chunk.char_end,
                json.dumps(chunk.section_path),
                chunk.token_count,
                chunk.chunk_type,
                chunk.boundary_confidence,
                json.dumps(chunk.manual_override) if chunk.manual_override else None,
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def insert_chunks_batch(self, chunks: list[Chunk]) -> list[int]:
        ids = []
        for chunk in chunks:
            ids.append(self.insert_chunk(chunk))
        return ids

    def get_chunk(self, chunk_id: int) -> Chunk | None:
        row = self.conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    def get_chunks_for_document(self, document_id: int) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY char_start",
            (document_id,),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_chunks_for_run(self, run_id: int) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE chunking_run_id = ? ORDER BY char_start",
            (run_id,),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def update_chunk(self, chunk: Chunk) -> None:
        self.conn.execute(
            """UPDATE chunks SET text = ?, char_start = ?, char_end = ?,
               section_path_json = ?, token_count = ?, chunk_type = ?,
               boundary_confidence = ?, manual_override_json = ?
               WHERE id = ?""",
            (
                chunk.text,
                chunk.char_start,
                chunk.char_end,
                json.dumps(chunk.section_path),
                chunk.token_count,
                chunk.chunk_type,
                chunk.boundary_confidence,
                json.dumps(chunk.manual_override) if chunk.manual_override else None,
                chunk.id,
            ),
        )
        self.conn.commit()

    def delete_chunks_for_run(self, run_id: int) -> None:
        self.conn.execute("DELETE FROM chunks WHERE chunking_run_id = ?", (run_id,))
        self.conn.commit()

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        override = row["manual_override_json"]
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            chunking_run_id=row["chunking_run_id"],
            text=row["text"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            section_path=json.loads(row["section_path_json"]),
            token_count=row["token_count"] or 0,
            chunk_type=row["chunk_type"],
            boundary_confidence=row["boundary_confidence"],
            manual_override=json.loads(override) if override else None,
        )

    # --- Labels ---

    def insert_label(self, label: Label) -> int:
        cur = self.conn.execute(
            """INSERT INTO labels
               (chunk_id, rubric_version_id, predicted_categories_json,
                confidence_json, rationale, knn_agreement, composite_confidence,
                llm_model, llm_run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                label.chunk_id,
                label.rubric_version_id,
                json.dumps(label.predicted_categories),
                json.dumps(label.confidence_per_category),
                label.rationale,
                label.knn_agreement,
                label.composite_confidence,
                label.llm_model,
                label.llm_run_id,
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_labels_for_chunk(self, chunk_id: int) -> list[Label]:
        rows = self.conn.execute(
            "SELECT * FROM labels WHERE chunk_id = ? ORDER BY created_at",
            (chunk_id,),
        ).fetchall()
        return [self._row_to_label(r) for r in rows]

    def get_all_labels(self, rubric_version_id: int | None = None) -> list[Label]:
        if rubric_version_id is not None:
            rows = self.conn.execute(
                "SELECT * FROM labels WHERE rubric_version_id = ?",
                (rubric_version_id,),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM labels").fetchall()
        return [self._row_to_label(r) for r in rows]

    def get_unlabeled_chunks(self, rubric_version_id: int) -> list[Chunk]:
        rows = self.conn.execute(
            """SELECT c.* FROM chunks c
               WHERE c.id NOT IN (
                   SELECT chunk_id FROM labels WHERE rubric_version_id = ?
               )
               ORDER BY c.char_start""",
            (rubric_version_id,),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_chunks_for_review(
        self, rubric_version_id: int, threshold: float
    ) -> list[tuple[Chunk, Label]]:
        rows = self.conn.execute(
            """SELECT c.*, l.id as l_id, l.chunk_id as l_chunk_id,
                      l.rubric_version_id as l_rubric_version_id,
                      l.predicted_categories_json, l.confidence_json,
                      l.rationale as l_rationale, l.knn_agreement,
                      l.composite_confidence, l.llm_model, l.llm_run_id,
                      l.created_at as l_created_at
               FROM chunks c
               JOIN labels l ON c.id = l.chunk_id
               WHERE l.rubric_version_id = ?
                 AND l.composite_confidence <= ?
                 AND l.id NOT IN (SELECT label_id FROM human_reviews)
               ORDER BY l.composite_confidence ASC""",
            (rubric_version_id, threshold),
        ).fetchall()
        results = []
        for r in rows:
            chunk = self._row_to_chunk(r)
            label = Label(
                id=r["l_id"],
                chunk_id=r["l_chunk_id"],
                rubric_version_id=r["l_rubric_version_id"],
                predicted_categories=json.loads(r["predicted_categories_json"]),
                confidence_per_category=json.loads(r["confidence_json"]),
                rationale=r["l_rationale"] or "",
                knn_agreement=r["knn_agreement"],
                composite_confidence=r["composite_confidence"] or 0.0,
                llm_model=r["llm_model"] or "",
                llm_run_id=r["llm_run_id"] or "",
                created_at=r["l_created_at"] or "",
            )
            results.append((chunk, label))
        return results

    def _row_to_label(self, row: sqlite3.Row) -> Label:
        return Label(
            id=row["id"],
            chunk_id=row["chunk_id"],
            rubric_version_id=row["rubric_version_id"],
            predicted_categories=json.loads(row["predicted_categories_json"]),
            confidence_per_category=json.loads(row["confidence_json"]),
            rationale=row["rationale"] or "",
            knn_agreement=row["knn_agreement"],
            composite_confidence=row["composite_confidence"] or 0.0,
            llm_model=row["llm_model"] or "",
            llm_run_id=row["llm_run_id"] or "",
            created_at=row["created_at"] or "",
        )

    # --- Human reviews ---

    def insert_review(self, review: HumanReview) -> int:
        cur = self.conn.execute(
            """INSERT INTO human_reviews
               (label_id, reviewer, action, final_categories_json, notes)
               VALUES (?, ?, ?, ?, ?)""",
            (
                review.label_id,
                review.reviewer,
                review.action,
                json.dumps(review.final_categories),
                review.notes,
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_reviews_for_label(self, label_id: int) -> list[HumanReview]:
        rows = self.conn.execute(
            "SELECT * FROM human_reviews WHERE label_id = ?", (label_id,)
        ).fetchall()
        return [self._row_to_review(r) for r in rows]

    def _row_to_review(self, row: sqlite3.Row) -> HumanReview:
        return HumanReview(
            id=row["id"],
            label_id=row["label_id"],
            reviewer=row["reviewer"],
            action=row["action"],
            final_categories=json.loads(row["final_categories_json"]),
            notes=row["notes"] or "",
            reviewed_at=row["reviewed_at"] or "",
        )

    # --- Audit events ---

    def insert_audit_event(self, event: AuditEvent) -> int:
        cur = self.conn.execute(
            """INSERT INTO audit_events
               (event_type, actor, payload_json, related_chunk_ids_json)
               VALUES (?, ?, ?, ?)""",
            (
                event.event_type,
                event.actor,
                json.dumps(event.payload),
                json.dumps(event.related_chunk_ids),
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_audit_events(
        self, event_type: str | None = None, limit: int = 100
    ) -> list[AuditEvent]:
        if event_type:
            rows = self.conn.execute(
                "SELECT * FROM audit_events WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM audit_events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_audit_event(r) for r in rows]

    def _row_to_audit_event(self, row: sqlite3.Row) -> AuditEvent:
        return AuditEvent(
            id=row["id"],
            event_type=row["event_type"],
            actor=row["actor"],
            timestamp=row["timestamp"] or "",
            payload=json.loads(row["payload_json"]),
            related_chunk_ids=json.loads(row["related_chunk_ids_json"]),
        )

    # --- Summary queries ---

    def get_labeling_summary(self, rubric_version_id: int) -> dict:
        """Get summary statistics for labeling results."""
        total = self.conn.execute(
            "SELECT COUNT(*) FROM labels WHERE rubric_version_id = ?",
            (rubric_version_id,),
        ).fetchone()[0]

        reviewed = self.conn.execute(
            """SELECT COUNT(*) FROM human_reviews hr
               JOIN labels l ON hr.label_id = l.id
               WHERE l.rubric_version_id = ?""",
            (rubric_version_id,),
        ).fetchone()[0]

        rows = self.conn.execute(
            "SELECT predicted_categories_json, composite_confidence FROM labels WHERE rubric_version_id = ?",
            (rubric_version_id,),
        ).fetchall()

        category_counts: dict[str, int] = {}
        confidences: list[float] = []
        for r in rows:
            cats = json.loads(r["predicted_categories_json"])
            for cat in cats:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            confidences.append(r["composite_confidence"] or 0.0)

        return {
            "total_labels": total,
            "reviewed": reviewed,
            "category_counts": category_counts,
            "confidences": confidences,
        }

    def get_document_label_status(self) -> list[dict]:
        """Get labeling status per document."""
        rows = self.conn.execute(
            """SELECT d.id, d.filename, d.status,
                      COUNT(DISTINCT c.id) as chunk_count,
                      COUNT(DISTINCT l.id) as label_count
               FROM documents d
               LEFT JOIN chunks c ON d.id = c.document_id
               LEFT JOIN labels l ON c.id = l.chunk_id
               GROUP BY d.id
               ORDER BY d.filename"""
        ).fetchall()
        return [
            {
                "document_id": r["id"],
                "filename": r["filename"],
                "status": r["status"],
                "chunk_count": r["chunk_count"],
                "label_count": r["label_count"],
            }
            for r in rows
        ]

    # --- Lifecycle ---

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
