"""PDF document converter using PyMuPDF."""

from __future__ import annotations

import statistics
from pathlib import Path

from lazylabeltext.core.converters import _register
from lazylabeltext.core.exceptions import DocumentParseError
from lazylabeltext.core.models import (
    ConvertedDocument,
    Heading,
    PageBoundary,
    SectionBoundary,
)


class PDFConverter:
    """Converts PDF files to internal representation using PyMuPDF (fitz)."""

    def convert(self, file_path: str) -> ConvertedDocument:
        path = Path(file_path)
        try:
            import fitz
        except ImportError as e:
            raise DocumentParseError(file_path, "PyMuPDF not installed") from e

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            raise DocumentParseError(file_path, str(e)) from e

        text_parts: list[str] = []
        pages: list[PageBoundary] = []
        all_spans: list[dict] = []
        char_offset = 0
        warnings: list[str] = []

        # First pass: collect all text and font info
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_start = char_offset

            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)[
                "blocks"
            ]
            page_text_parts: list[str] = []

            for block in blocks:
                if block.get("type") != 0:  # text blocks only
                    continue
                for line in block.get("lines", []):
                    line_text = ""
                    line_sizes: list[float] = []
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        line_text += span_text
                        size = span.get("size", 12.0)
                        line_sizes.append(size)
                        flags = span.get("flags", 0)
                        all_spans.append(
                            {
                                "text": span_text,
                                "size": size,
                                "bold": bool(flags & 2**4),
                                "char_start": char_offset
                                + len("".join(page_text_parts))
                                + len(line_text)
                                - len(span_text),
                            }
                        )

                    if line_text.strip():
                        page_text_parts.append(line_text.strip())

            page_text = "\n".join(page_text_parts)
            if page_text:
                text_parts.append(page_text)
                page_end = char_offset + len(page_text)
                pages.append(
                    PageBoundary(
                        page_number=page_num + 1,
                        char_start=page_start,
                        char_end=page_end,
                    )
                )
                char_offset = page_end + 1  # +1 for page separator newline

        full_text = "\n".join(text_parts)

        # Detect headings via font size analysis
        headings = self._detect_headings(full_text, all_spans)
        sections = self._build_sections(headings, len(full_text))

        # Extract metadata
        metadata: dict = {"source": str(path), "page_count": len(doc)}
        pdf_meta = doc.metadata
        if pdf_meta:
            for key in ("title", "author", "subject", "creator", "creationDate"):
                if pdf_meta.get(key):
                    metadata[key] = pdf_meta[key]

        if not full_text.strip():
            warnings.append("No extractable text found (may be a scanned PDF)")

        doc.close()

        return ConvertedDocument(
            filename=path.name,
            format="pdf",
            full_text=full_text,
            headings=headings,
            pages=pages,
            sections=sections,
            metadata=metadata,
            status="parsed" if not warnings else "warnings",
            warnings=warnings,
        )

    def _detect_headings(self, text: str, spans: list[dict]) -> list[Heading]:
        """Detect headings by analyzing font size patterns."""
        if not spans:
            return []

        sizes = [s["size"] for s in spans if s["text"].strip()]
        if not sizes:
            return []

        median_size = statistics.median(sizes)
        max_size = max(sizes)

        if max_size <= median_size * 1.1:
            return []  # No size variation, no headings

        # Group spans into lines
        headings: list[Heading] = []
        lines = text.split("\n")
        char_offset = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                char_offset += len(line) + 1
                continue

            # Find spans that belong to this line
            line_spans = [
                s for s in spans if s["text"].strip() and s["text"].strip() in stripped
            ]

            if line_spans:
                avg_size = statistics.mean([s["size"] for s in line_spans])
                is_bold = any(s["bold"] for s in line_spans)
                is_short = len(stripped) < 120

                if avg_size > median_size * 1.15 and is_short:
                    # Determine heading level from font size
                    size_ratio = avg_size / median_size
                    if size_ratio > 1.6:
                        level = 1
                    elif size_ratio > 1.3:
                        level = 2
                    else:
                        level = 3

                    headings.append(
                        Heading(
                            level=level,
                            text=stripped,
                            char_start=char_offset,
                            char_end=char_offset + len(stripped),
                        )
                    )
                elif is_bold and is_short and avg_size >= median_size:
                    headings.append(
                        Heading(
                            level=3,
                            text=stripped,
                            char_start=char_offset,
                            char_end=char_offset + len(stripped),
                        )
                    )

            char_offset += len(line) + 1

        return headings

    def _build_sections(
        self, headings: list[Heading], text_length: int
    ) -> list[SectionBoundary]:
        if not headings:
            return [
                SectionBoundary(section_path=[], char_start=0, char_end=text_length)
            ]

        sections = []
        path_stack: list[str] = []

        for i, heading in enumerate(headings):
            while len(path_stack) >= heading.level:
                path_stack.pop()
            path_stack.append(heading.text)

            end = headings[i + 1].char_start if i + 1 < len(headings) else text_length
            sections.append(
                SectionBoundary(
                    section_path=list(path_stack),
                    char_start=heading.char_start,
                    char_end=end,
                )
            )
        return sections

    def supported_extensions(self) -> set[str]:
        return {".pdf"}


_register({".pdf"}, PDFConverter())
