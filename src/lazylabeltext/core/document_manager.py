"""Document loading and conversion management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from lazylabeltext.core.converters import convert_file, supported_extensions
from lazylabeltext.core.database import Database
from lazylabeltext.core.doc_map import DocMap, DocMapResolveResult
from lazylabeltext.core.exceptions import DocumentParseError
from lazylabeltext.core.models import ConvertedDocument
from lazylabeltext.utils.file_hash import hash_file

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

    def find_supported_files_from_docmap(
        self, docmap_path: str
    ) -> tuple[list[str], DocMapResolveResult]:
        """Resolve a doc map to supported source paths.

        Returns ``(paths, result)`` where ``result`` carries the unmatched
        patterns so callers can surface a warning toast. Raises
        ``DocMapError`` if the file can't be read or parsed.
        """
        dm = DocMap.load(docmap_path)
        result = dm.resolve(supported_extensions())
        return result.files, result

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
        """Convert a single file and store in database, deduped by content hash.

        Identity is the SHA256 of the source bytes — moving or copying a
        file doesn't create a duplicate row, and two distinct files that
        happen to share a basename are correctly recognised as separate.

        If the same content already exists in the project under a different
        path/name (e.g. the file was moved between sessions), the existing
        row is returned and its source path is updated to the new location.

        Legacy rows (ingested before content-hash dedupe) have an empty
        ``source_hash`` and are matched by filename as a fallback; the hash
        is then backfilled so subsequent ingests use the strong identity.
        """
        path = Path(file_path)

        # Hash the source bytes up front. If the file is unreadable, fall
        # back to filename-based dedupe so we don't drop conversions we
        # used to handle correctly.
        try:
            src_hash = hash_file(file_path)
        except OSError as e:
            logger.warning("Could not hash %s for dedupe: %s", file_path, e)
            src_hash = ""

        if src_hash:
            existing = self.db.get_document_by_hash(src_hash)
            if existing is not None and existing.id is not None:
                # Same content already in DB. Update path/filename if the
                # file was moved or renamed; otherwise it's an idempotent
                # re-ingest.
                prior_source = existing.metadata.get("source", "")
                if prior_source != str(path) or existing.filename != path.name:
                    logger.info(
                        "Doc already in project at new path; updating "
                        "%s → %s", prior_source, path,
                    )
                    self.db.update_document_source(
                        existing.id,
                        source_path=str(path),
                        filename=path.name,
                    )
                    existing.metadata["source"] = str(path)
                    existing.filename = path.name
                return existing

        # Legacy / no-hash fallback: dedupe by filename if a row exists with
        # an empty source_hash. Backfill the hash so future runs use the
        # strong identity.
        legacy = self.db.get_document_by_filename(path.name)
        if legacy is not None and not (legacy.source_hash or ""):
            if src_hash and legacy.id is not None:
                self.db.set_document_source_hash(legacy.id, src_hash)
                legacy.source_hash = src_hash
            return legacy

        doc = convert_file(file_path, **self._conversion_flags())
        doc.metadata.setdefault("source", str(path))
        doc.source_hash = src_hash
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
        # Source content may have changed — recompute the hash so dedupe
        # stays correct against the *current* bytes.
        try:
            doc.source_hash = hash_file(source)
        except OSError:
            doc.source_hash = existing.source_hash or ""
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
