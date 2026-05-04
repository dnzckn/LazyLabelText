"""Structural chunker: splits on headings, enforces token bounds."""

from __future__ import annotations

from lazylabeltext.core.chunkers import _register
from lazylabeltext.core.models import Chunk, ConvertedDocument
from lazylabeltext.utils.token_counter import count_tokens


class StructuralChunker:
    """Chunks text by heading boundaries with token count constraints."""

    @property
    def name(self) -> str:
        return "structural"

    def chunk(self, document: ConvertedDocument, params: dict) -> list[Chunk]:
        min_tokens = params.get("min_tokens", 50)
        max_tokens = params.get("max_tokens", 800)
        heading_levels = set(params.get("heading_split_levels", [1, 2, 3]))

        text = document.full_text
        if not text.strip():
            return []

        # Find split points from headings at specified levels
        split_points = [0]
        for heading in document.headings:
            if (
                heading.level in heading_levels
                and heading.char_start not in split_points
            ):
                split_points.append(heading.char_start)
        split_points.append(len(text))
        split_points.sort()

        # Create initial chunks from heading splits
        raw_chunks: list[tuple[int, int, list[str]]] = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue

            # Determine section path
            section_path = self._get_section_path(document, start)
            raw_chunks.append((start, end, section_path))

        # Enforce token constraints
        final_chunks: list[Chunk] = []
        doc_id = document.id or 0

        for start, end, section_path in raw_chunks:
            chunk_text = text[start:end].strip()
            tokens = count_tokens(chunk_text)

            if tokens <= max_tokens:
                # Chunk is within bounds
                if tokens < min_tokens and final_chunks:
                    # Merge with previous chunk if too small
                    prev = final_chunks[-1]
                    merged_text = prev.text + "\n\n" + chunk_text
                    prev.text = merged_text
                    prev.char_end = end
                    prev.token_count = count_tokens(merged_text)
                else:
                    final_chunks.append(
                        Chunk(
                            document_id=doc_id,
                            text=chunk_text,
                            char_start=start,
                            char_end=end,
                            section_path=section_path,
                            token_count=tokens,
                        )
                    )
            else:
                # Split oversized chunks at paragraph boundaries
                sub_chunks = self._split_large_chunk(
                    chunk_text, start, max_tokens, section_path, doc_id
                )
                final_chunks.extend(sub_chunks)

        return final_chunks

    def _split_large_chunk(
        self,
        text: str,
        base_offset: int,
        max_tokens: int,
        section_path: list[str],
        doc_id: int,
    ) -> list[Chunk]:
        """Split an oversized chunk at paragraph boundaries."""
        paragraphs = text.split("\n\n")
        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_start = base_offset
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = count_tokens(para)

            if current_tokens + para_tokens > max_tokens and current_parts:
                # Emit current chunk
                chunk_text = "\n\n".join(current_parts)
                chunks.append(
                    Chunk(
                        document_id=doc_id,
                        text=chunk_text,
                        char_start=current_start,
                        char_end=current_start + len(chunk_text),
                        section_path=section_path,
                        token_count=current_tokens,
                    )
                )
                current_start = current_start + len(chunk_text) + 2
                current_parts = []
                current_tokens = 0

            current_parts.append(para)
            current_tokens += para_tokens

        # Emit remaining
        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(
                Chunk(
                    document_id=doc_id,
                    text=chunk_text,
                    char_start=current_start,
                    char_end=current_start + len(chunk_text),
                    section_path=section_path,
                    token_count=current_tokens,
                )
            )

        return chunks

    def _get_section_path(
        self, document: ConvertedDocument, char_pos: int
    ) -> list[str]:
        """Find the section path for a character position."""
        for section in reversed(document.sections):
            if section.char_start <= char_pos < section.char_end:
                return section.section_path
        return []


_register("structural", StructuralChunker())
