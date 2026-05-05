"""Document loading and conversion management."""

from __future__ import annotations

import logging
from pathlib import Path

from lazylabeltext.core.converters import convert_file, supported_extensions
from lazylabeltext.core.database import Database
from lazylabeltext.core.exceptions import DocumentParseError
from lazylabeltext.core.models import ConvertedDocument

logger = logging.getLogger("lazylabeltext")


class DocumentManager:
    """Manages document loading, conversion, and storage."""

    def __init__(self, database: Database) -> None:
        self.db = database

    def load_folder(self, folder_path: str) -> list[ConvertedDocument]:
        """Walk a folder, convert supported files, store in database."""
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Not a directory: {folder_path}")

        supported = supported_extensions()
        documents: list[ConvertedDocument] = []

        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in supported:
                try:
                    doc = self.load_single(str(path))
                    documents.append(doc)
                except DocumentParseError as e:
                    logger.warning("Failed to convert %s: %s", path.name, e)
                    # Store failed document for visibility
                    failed_doc = ConvertedDocument(
                        filename=path.name,
                        format=path.suffix.lower().lstrip("."),
                        status="failed",
                        warnings=[str(e)],
                        metadata={"source": str(path)},
                    )
                    failed_doc.id = self.db.insert_document(failed_doc)
                    documents.append(failed_doc)

        return documents

    def load_single(self, file_path: str) -> ConvertedDocument:
        """Convert a single file and store in database (idempotent on filename).

        If a document with this filename already exists in the project, return
        the existing row — re-opening a folder must not duplicate documents
        and orphan their chunks/labels under stale IDs.
        """
        path = Path(file_path)
        existing = self.db.get_document_by_filename(path.name)
        if existing is not None:
            logger.debug(
                "Document already in project: %s (id=%d)", path.name, existing.id or 0
            )
            return existing

        doc = convert_file(file_path)
        doc.id = self.db.insert_document(doc)
        logger.info(
            "Loaded %s: %d chars, %d headings",
            doc.filename,
            len(doc.full_text),
            len(doc.headings),
        )
        return doc

    def get_document(self, doc_id: int) -> ConvertedDocument | None:
        return self.db.get_document(doc_id)

    def get_all_documents(self) -> list[ConvertedDocument]:
        return self.db.get_all_documents()
