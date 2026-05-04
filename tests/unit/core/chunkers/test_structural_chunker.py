"""Tests for structural chunker."""

from __future__ import annotations

from lazylabeltext.core.chunkers.structural_chunker import StructuralChunker
from lazylabeltext.core.models import ConvertedDocument, Heading


class TestStructuralChunker:
    def setup_method(self):
        self.chunker = StructuralChunker()

    def test_empty_document(self):
        doc = ConvertedDocument(full_text="", headings=[])
        chunks = self.chunker.chunk(doc, {})
        assert chunks == []

    def test_single_section(self):
        doc = ConvertedDocument(
            full_text="This is some body text with enough words to make a chunk.",
            headings=[],
        )
        chunks = self.chunker.chunk(doc, {"min_tokens": 1, "max_tokens": 1000})
        assert len(chunks) == 1
        assert chunks[0].text.strip() == doc.full_text.strip()

    def test_splits_on_headings(self):
        text = "# Intro\n\nIntro text here.\n\n# Details\n\nDetails text here."
        doc = ConvertedDocument(
            full_text=text,
            headings=[
                Heading(level=1, text="Intro", char_start=0, char_end=7),
                Heading(level=1, text="Details", char_start=27, char_end=36),
            ],
        )
        chunks = self.chunker.chunk(
            doc, {"heading_split_levels": [1], "min_tokens": 1, "max_tokens": 1000}
        )
        assert len(chunks) == 2

    def test_respects_heading_levels(self):
        text = "# H1\n\nText.\n\n## H2\n\nMore text.\n\n### H3\n\nEven more."
        doc = ConvertedDocument(
            full_text=text,
            headings=[
                Heading(level=1, text="H1", char_start=0, char_end=4),
                Heading(level=2, text="H2", char_start=14, char_end=19),
                Heading(level=3, text="H3", char_start=32, char_end=38),
            ],
        )
        # Only split on H1
        chunks = self.chunker.chunk(
            doc, {"heading_split_levels": [1], "min_tokens": 1, "max_tokens": 1000}
        )
        assert len(chunks) == 1  # Only one H1 at start, no split point after

    def test_merges_small_chunks(self):
        text = "# A\n\nHi\n\n# B\n\nLonger text that is substantial."
        doc = ConvertedDocument(
            full_text=text,
            headings=[
                Heading(level=1, text="A", char_start=0, char_end=3),
                Heading(level=1, text="B", char_start=9, char_end=12),
            ],
        )
        chunks = self.chunker.chunk(
            doc, {"heading_split_levels": [1], "min_tokens": 20, "max_tokens": 1000}
        )
        # "Hi" is too small, merged with next
        assert len(chunks) <= 2

    def test_name_property(self):
        assert self.chunker.name == "structural"
