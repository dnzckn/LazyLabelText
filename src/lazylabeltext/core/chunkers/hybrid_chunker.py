"""Hybrid chunker: structural pass, then semantic split on oversized chunks."""

from __future__ import annotations

from lazylabeltext.core.chunkers import _register
from lazylabeltext.core.chunkers.semantic_chunker import chunk_text_semantic
from lazylabeltext.core.chunkers.structural_chunker import StructuralChunker
from lazylabeltext.core.exceptions import ChunkOperationError
from lazylabeltext.core.models import Chunk, ConvertedDocument


class HybridChunker:
    """Structural-first chunker; falls back to semantic on oversized chunks."""

    def __init__(self) -> None:
        self._structural = StructuralChunker()

    @property
    def name(self) -> str:
        return "hybrid"

    def chunk(self, document: ConvertedDocument, params: dict) -> list[Chunk]:
        provider = params.get("_embedding_provider")
        if provider is None:
            raise ChunkOperationError(
                "chunk",
                "hybrid chunker requires an embedding provider; configure one in "
                "Provider Settings.",
            )

        max_tokens = params.get("max_tokens", 800)
        min_tokens = params.get("min_tokens", 50)
        threshold = params.get("similarity_threshold", 0.5)

        structural_chunks = self._structural.chunk(document, params)

        result: list[Chunk] = []
        for chunk in structural_chunks:
            if (chunk.token_count or 0) <= max_tokens:
                result.append(chunk)
                continue

            sub_chunks = chunk_text_semantic(
                text=chunk.text,
                base_offset=chunk.char_start,
                embedding_provider=provider,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                threshold=threshold,
                section_path=chunk.section_path,
                doc_id=chunk.document_id,
            )
            result.extend(sub_chunks if sub_chunks else [chunk])

        return result


_register("hybrid", HybridChunker())
