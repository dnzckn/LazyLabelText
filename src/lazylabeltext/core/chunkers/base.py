"""Base chunker protocol."""

from __future__ import annotations

from typing import Protocol

from lazylabeltext.core.models import Chunk, ConvertedDocument


class Chunker(Protocol):
    """Protocol for text chunking strategies."""

    def chunk(self, document: ConvertedDocument, params: dict) -> list[Chunk]: ...

    @property
    def name(self) -> str: ...
