"""Document loading and conversion management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from lazylabeltext.core.converters import convert_file, supported_extensions
from lazylabeltext.core.database import Database
from lazylabeltext.core.exceptions import DocumentParseError
from lazylabeltext.core.models import ConvertedDocument

if TYPE_CHECKING:
    from lazylabeltext.config.settings import Settings

logger = logging.getLogger("lazylabeltext")


class DocumentManager:
    """Manages document loading, conversion, and storage."""

    def __init__(
        self, database: Database, settings: Settings | None = None
    ) -> None:
        self.db = database
        self.settings = settings

    def _conversion_flags(self) -> dict:
        if self.settings is None:
            return {"high_fidelity": False, "ocr_enabled": False}
        return {
            "high_fidelity": getattr(
                self.settings, "use_high_fidelity_conversion", False
            ),
            "ocr_enabled": getattr(self.settings, "high_fidelity_ocr", False),
        }

    def find_supported_files(self, folder_path: str) -> list[str]:
        """List supported files in a folder (recursive). Used by background workers."""
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Not a directory: {folder_path}")
        supported = supported_extensions()
        return [
            str(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file() and p.suffix.lower() in supported
        ]

    def record_failed(self, file_path: str, error: str) -> ConvertedDocument:
        """Insert a stub 'failed' row so the user can see what didn't convert."""
        path = Path(file_path)
        existing = self.db.get_document_by_filename(path.name)
        if existing is not None:
            return existing
        failed_doc = ConvertedDocument(
            filename=path.name,
            format=path.suffix.lower().lstrip("."),
            status="failed",
            warnings=[error],
            metadata={"source": str(path)},
        )
        failed_doc.id = self.db.insert_document(failed_doc)
        return failed_doc

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

        doc = convert_file(file_path, **self._conversion_flags())
        doc.id = self.db.insert_document(doc)
        logger.info(
            "Loaded %s: %d chars, %d headings",
            doc.filename,
            len(doc.full_text),
            len(doc.headings),
        )
        return doc

    def reconvert(self, doc_id: int) -> ConvertedDocument | None:
        """Re-run conversion for an existing document using current settings.

        Updates the document row in place — preserving the id so any UI state
        elsewhere (chunk/label/results modes) that still holds this doc_id
        keeps resolving. Chunks/labels/reviews are cascade-deleted because
        the new text invalidates them. Returns None if the original source
        file is no longer reachable.
        """
        existing = self.db.get_document(doc_id)
        if existing is None:
            return None

        source = existing.metadata.get("source", "")
        if not source or not Path(source).is_file():
            logger.warning(
                "Cannot reconvert %s: source file missing (%s)",
                existing.filename,
                source,
            )
            return None

        doc = convert_file(source, **self._conversion_flags())
        self.db.replace_document_content(doc_id, doc)
        doc.id = doc_id
        logger.info(
            "Reconverted %s: %d chars, %d headings",
            doc.filename,
            len(doc.full_text),
            len(doc.headings),
        )
        return doc

    def get_document(self, doc_id: int) -> ConvertedDocument | None:
        return self.db.get_document(doc_id)

    def get_all_documents(self) -> list[ConvertedDocument]:
        return self.db.get_all_documents()
