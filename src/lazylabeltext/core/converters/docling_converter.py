"""High-fidelity converter for PDF and DOCX using docling.

Docling parses documents into a structured tree (sections, tables, figures,
text) and exports clean markdown. This converter is opt-in: install via
``pip install lazylabeltext[high-fidelity]``. The first conversion will
download ~1–2 GB of layout/OCR models.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from pathlib import Path

from lazylabeltext.core.exceptions import DocumentParseError
from lazylabeltext.core.models import (
    ConvertedDocument,
    Heading,
    PageBoundary,
    SectionBoundary,
)

logger = logging.getLogger("lazylabeltext")

# Module-level singletons keyed by (ocr_enabled,) — building a DocumentConverter
# loads the layout/table-structure models, which is the slow part. Reusing
# them across calls is the difference between 30s and 5s per page.
_converter_cache: dict[tuple[bool], object] = {}
_converter_lock = threading.Lock()


def _get_converter(ocr_enabled: bool) -> object:
    key = (ocr_enabled,)
    with _converter_lock:
        cached = _converter_cache.get(key)
        if cached is not None:
            return cached

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr = ocr_enabled
        pdf_opts.do_table_structure = True

        conv = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
            }
        )
        _converter_cache[key] = conv
        return conv


class DoclingConverter:
    """Converts PDF/DOCX via docling, preserving tables and figure references."""

    SUPPORTED = {".pdf", ".docx"}

    def convert(
        self, file_path: str, *, ocr_enabled: bool = False
    ) -> ConvertedDocument:
        path = Path(file_path)

        try:
            converter = _get_converter(ocr_enabled)
        except ImportError as e:
            raise DocumentParseError(
                file_path,
                "docling not installed — run: pip install lazylabeltext[high-fidelity]",
            ) from e
        except Exception as e:
            raise DocumentParseError(
                file_path, f"failed to initialize docling: {e}"
            ) from e

        try:
            result = converter.convert(str(path))  # type: ignore[attr-defined]
        except Exception as e:
            raise DocumentParseError(file_path, f"docling failed: {e}") from e

        dl_doc = result.document
        warnings: list[str] = []

        full_text = dl_doc.export_to_markdown()

        headings = self._extract_headings(dl_doc, full_text)
        pages = self._extract_pages(dl_doc, full_text)
        sections = self._build_sections(headings, len(full_text))

        metadata: dict = {
            "source": str(path),
            "backend": "docling",
            "ocr": ocr_enabled,
        }
        tables = self._collect_tables(dl_doc)
        if tables:
            metadata["tables"] = tables
        figures = self._collect_figures(dl_doc)
        if figures:
            metadata["figures"] = figures
        if pages:
            metadata["page_count"] = len(pages)

        if not full_text.strip():
            warnings.append("No extractable content found")

        fmt = path.suffix.lower().lstrip(".") or "unknown"
        return ConvertedDocument(
            filename=path.name,
            format=fmt,
            full_text=full_text,
            headings=headings,
            pages=pages,
            sections=sections,
            metadata=metadata,
            status="parsed" if not warnings else "warnings",
            warnings=warnings,
        )

    def _extract_headings(self, dl_doc: object, full_text: str) -> list[Heading]:
        """Walk docling section headers and locate them in the markdown export.

        Docling emits headings as ``# text`` / ``## text`` lines; we scan the
        markdown line by line so char offsets line up with full_text exactly.
        """
        headings: list[Heading] = []
        offset = 0
        for line in full_text.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                hashes = 0
                for ch in stripped:
                    if ch == "#":
                        hashes += 1
                    else:
                        break
                if 1 <= hashes <= 6 and (
                    len(stripped) == hashes or stripped[hashes] == " "
                ):
                    text = stripped[hashes:].strip()
                    if text:
                        # char_start points at the first '#' so highlighters
                        # can decide whether to include the marker.
                        leading_ws = len(line) - len(stripped)
                        start = offset + leading_ws
                        headings.append(
                            Heading(
                                level=hashes,
                                text=text,
                                char_start=start,
                                char_end=start + len(stripped),
                            )
                        )
            offset += len(line) + 1  # +1 for the newline split() consumed
        return headings

    def _extract_pages(self, dl_doc: object, full_text: str) -> list[PageBoundary]:
        """Approximate page boundaries from docling page metadata.

        Docling's items carry provenance (page_no), but mapping each item back
        into the exported markdown text is fragile. We fall back to even
        splits across the markdown when page count is known — good enough for
        navigation/highlighting and matches PyMuPDF's char-offset semantics.
        """
        pages_attr = getattr(dl_doc, "pages", None)
        if not pages_attr:
            return []
        try:
            page_count = len(pages_attr)
        except TypeError:
            page_count = sum(1 for _ in pages_attr)
        if page_count <= 0:
            return []

        text_len = len(full_text)
        boundaries: list[PageBoundary] = []
        for i in range(page_count):
            start = (text_len * i) // page_count
            end = (text_len * (i + 1)) // page_count
            boundaries.append(
                PageBoundary(page_number=i + 1, char_start=start, char_end=end)
            )
        return boundaries

    def _collect_tables(self, dl_doc: object) -> list[dict]:
        out: list[dict] = []
        tables = getattr(dl_doc, "tables", None) or []
        for t in tables:
            entry: dict = {}
            prov = getattr(t, "prov", None)
            if prov:
                first = prov[0] if hasattr(prov, "__getitem__") else next(iter(prov))
                page_no = getattr(first, "page_no", None)
                if page_no is not None:
                    entry["page"] = page_no
            caption = getattr(t, "caption_text", None)
            if callable(caption):
                with contextlib.suppress(Exception):
                    entry["caption"] = caption(dl_doc)
            out.append(entry)
        return out

    def _collect_figures(self, dl_doc: object) -> list[dict]:
        out: list[dict] = []
        pictures = getattr(dl_doc, "pictures", None) or []
        for p in pictures:
            entry: dict = {}
            prov = getattr(p, "prov", None)
            if prov:
                first = prov[0] if hasattr(prov, "__getitem__") else next(iter(prov))
                page_no = getattr(first, "page_no", None)
                if page_no is not None:
                    entry["page"] = page_no
            out.append(entry)
        return out

    def _build_sections(
        self, headings: list[Heading], text_length: int
    ) -> list[SectionBoundary]:
        if not headings:
            return [
                SectionBoundary(section_path=[], char_start=0, char_end=text_length)
            ]
        sections: list[SectionBoundary] = []
        path_stack: list[str] = []
        for i, h in enumerate(headings):
            while len(path_stack) >= h.level:
                path_stack.pop()
            path_stack.append(h.text)
            end = headings[i + 1].char_start if i + 1 < len(headings) else text_length
            sections.append(
                SectionBoundary(
                    section_path=list(path_stack),
                    char_start=h.char_start,
                    char_end=end,
                )
            )
        return sections

    def supported_extensions(self) -> set[str]:
        return set(self.SUPPORTED)
