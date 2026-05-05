"""Chunker registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lazylabeltext.core.chunkers.base import Chunker
    from lazylabeltext.core.models import Chunk, ConvertedDocument

from lazylabeltext.core.exceptions import ChunkOperationError

CHUNKERS: dict[str, Chunker] = {}


def _register(name: str, chunker: Chunker) -> None:
    """Register a chunking strategy."""
    CHUNKERS[name] = chunker


def chunk_document(
    document: ConvertedDocument, strategy: str, params: dict
) -> list[Chunk]:
    """Chunk a document using the named strategy."""
    chunker = CHUNKERS.get(strategy)
    if chunker is None:
        raise ChunkOperationError(
            "chunk", f"Unknown strategy: {strategy}. Available: {list(CHUNKERS.keys())}"
        )
    return chunker.chunk(document, params)


def available_strategies() -> list[str]:
    """Return names of all registered chunking strategies."""
    return list(CHUNKERS.keys())


# Import submodules to trigger registration
from lazylabeltext.core.chunkers import structural_chunker  # noqa: E402, F401
from lazylabeltext.core.chunkers import semantic_chunker  # noqa: E402, F401
from lazylabeltext.core.chunkers import hybrid_chunker  # noqa: E402, F401
from lazylabeltext.core.chunkers import llm_chunker  # noqa: E402, F401
