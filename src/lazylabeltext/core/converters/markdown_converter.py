"""Markdown document converter."""

from __future__ import annotations

import re
from pathlib import Path

from lazylabeltext.core.converters import _register
from lazylabeltext.core.exceptions import DocumentParseError
from lazylabeltext.core.models import ConvertedDocument, Heading, SectionBoundary

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class MarkdownConverter:
    """Converts Markdown files to internal representation."""

    def convert(self, file_path: str) -> ConvertedDocument:
        path = Path(file_path)
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception as e:
            raise DocumentParseError(file_path, str(e)) from e

        headings = self._extract_headings(raw)
        sections = self._build_sections(headings, len(raw))

        # Strip markdown syntax for clean text
        clean = self._strip_markdown(raw)

        return ConvertedDocument(
            filename=path.name,
            format="md",
            full_text=raw,  # Keep raw for overlay display
            headings=headings,
            pages=[],
            sections=sections,
            metadata={"source": str(path), "clean_text_length": len(clean)},
            status="parsed",
        )

    def _extract_headings(self, text: str) -> list[Heading]:
        headings = []
        for match in _HEADING_RE.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append(
                Heading(
                    level=level,
                    text=title,
                    char_start=match.start(),
                    char_end=match.end(),
                )
            )
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
            # Trim stack to parent level
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

    def _strip_markdown(self, text: str) -> str:
        """Remove markdown syntax for clean text output."""
        # Remove headings markers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)
        # Remove links
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
        # Remove code blocks
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"`(.+?)`", r"\1", text)
        return text.strip()

    def supported_extensions(self) -> set[str]:
        return {".md", ".markdown"}


_register({".md", ".markdown"}, MarkdownConverter())
