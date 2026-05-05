"""Semantic chunker: splits at embedding similarity drops."""

from __future__ import annotations

import re

import numpy as np

from lazylabeltext.core.chunkers import _register
from lazylabeltext.core.exceptions import ChunkOperationError
from lazylabeltext.core.models import Chunk, ConvertedDocument
from lazylabeltext.utils.token_counter import count_tokens

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])|\n+")


def split_into_sentences(text: str) -> list[tuple[int, int, str]]:
    """Split text into sentence-like spans. Returns (start, end, text) tuples."""
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        end = match.start()
        segment = text[cursor:end]
        if segment.strip():
            spans.append((cursor, end, segment))
        cursor = match.end()
    if cursor < len(text):
        tail = text[cursor:]
        if tail.strip():
            spans.append((cursor, len(text), tail))
    return spans


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def chunk_text_semantic(
    text: str,
    base_offset: int,
    embedding_provider,
    min_tokens: int,
    max_tokens: int,
    threshold: float,
    section_path: list[str],
    doc_id: int,
) -> list[Chunk]:
    """Sentence-segment, embed, then group by similarity drops."""
    spans = split_into_sentences(text)
    if not spans:
        return []

    # Trivial case: single sentence, just emit.
    if len(spans) == 1:
        s, e, st = spans[0]
        return [
            Chunk(
                document_id=doc_id,
                text=st.strip(),
                char_start=base_offset + s,
                char_end=base_offset + e,
                section_path=section_path,
                token_count=count_tokens(st),
                boundary_confidence=1.0,
            )
        ]

    sentences = [st for _, _, st in spans]
    embeddings = embedding_provider.encode(sentences)

    chunks: list[Chunk] = []
    cur_start_idx = 0
    cur_tokens = count_tokens(sentences[0])

    for i in range(1, len(spans)):
        sim = _cosine(embeddings[i - 1], embeddings[i])
        sent_tokens = count_tokens(sentences[i])

        # Force-split if adding next sentence would exceed max_tokens.
        force_split = cur_tokens + sent_tokens > max_tokens
        # Soft split if similarity drops AND we have enough content.
        soft_split = sim < threshold and cur_tokens >= min_tokens

        if force_split or soft_split:
            start_char = base_offset + spans[cur_start_idx][0]
            end_char = base_offset + spans[i - 1][1]
            chunk_text = text[spans[cur_start_idx][0] : spans[i - 1][1]].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        document_id=doc_id,
                        text=chunk_text,
                        char_start=start_char,
                        char_end=end_char,
                        section_path=section_path,
                        token_count=count_tokens(chunk_text),
                        boundary_confidence=round(1.0 - sim, 3),
                    )
                )
            cur_start_idx = i
            cur_tokens = sent_tokens
        else:
            cur_tokens += sent_tokens

    # Emit trailing chunk
    start_char = base_offset + spans[cur_start_idx][0]
    end_char = base_offset + spans[-1][1]
    chunk_text = text[spans[cur_start_idx][0] : spans[-1][1]].strip()
    if chunk_text:
        chunks.append(
            Chunk(
                document_id=doc_id,
                text=chunk_text,
                char_start=start_char,
                char_end=end_char,
                section_path=section_path,
                token_count=count_tokens(chunk_text),
                boundary_confidence=1.0,
            )
        )

    return chunks


def _get_section_path(document: ConvertedDocument, char_pos: int) -> list[str]:
    for section in reversed(document.sections):
        if section.char_start <= char_pos < section.char_end:
            return section.section_path
    return []


class SemanticChunker:
    """Sentence-level chunker using embedding similarity drops."""

    @property
    def name(self) -> str:
        return "semantic"

    def chunk(self, document: ConvertedDocument, params: dict) -> list[Chunk]:
        provider = params.get("_embedding_provider")
        if provider is None:
            raise ChunkOperationError(
                "chunk",
                "semantic chunker requires an embedding provider; configure one in "
                "Provider Settings.",
            )

        text = document.full_text
        if not text.strip():
            return []

        return chunk_text_semantic(
            text=text,
            base_offset=0,
            embedding_provider=provider,
            min_tokens=params.get("min_tokens", 50),
            max_tokens=params.get("max_tokens", 800),
            threshold=params.get("similarity_threshold", 0.5),
            section_path=_get_section_path(document, 0),
            doc_id=document.id or 0,
        )


_register("semantic", SemanticChunker())
