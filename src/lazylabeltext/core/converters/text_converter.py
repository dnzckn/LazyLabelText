"""Plain text document converter."""

from __future__ import annotations

from pathlib import Path

from lazylabeltext.core.converters import _register
from lazylabeltext.core.exceptions import DocumentParseError
from lazylabeltext.core.models import ConvertedDocument, SectionBoundary


class TextConverter:
    """Converts plain text files to internal representation."""

    def convert(self, file_path: str) -> ConvertedDocument:
        path = Path(file_path)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="latin-1")
            except Exception as e:
                raise DocumentParseError(file_path, str(e)) from e

        return ConvertedDocument(
            filename=path.name,
            format="txt",
            full_text=text,
            headings=[],
            pages=[],
            sections=[
                SectionBoundary(
                    section_path=[path.stem],
                    char_start=0,
                    char_end=len(text),
                )
            ],
            metadata={"source": str(path)},
            status="parsed",
        )

    def supported_extensions(self) -> set[str]:
        return {".txt", ".text"}


_register({".txt", ".text"}, TextConverter())
