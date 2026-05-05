"""Protocol definitions for loose coupling between components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from lazylabeltext.core.models import (
        Category,
        Chunk,
        ClassificationResult,
        ConvertedDocument,
    )


@runtime_checkable
class ConverterProtocol(Protocol):
    """Protocol for document format converters."""

    def convert(self, file_path: str) -> ConvertedDocument: ...

    def supported_extensions(self) -> set[str]: ...


@runtime_checkable
class ChunkerProtocol(Protocol):
    """Protocol for text chunking strategies."""

    def chunk(self, document: ConvertedDocument, params: dict) -> list[Chunk]: ...

    @property
    def name(self) -> str: ...


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """Protocol for LLM classification providers."""

    def classify(
        self, chunk_text: str, categories: list[Category]
    ) -> ClassificationResult: ...

    def complete(self, prompt: str, max_tokens: int = 4096) -> str: ...


@runtime_checkable
class EmbeddingProviderProtocol(Protocol):
    """Protocol for text embedding providers."""

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def encode_one(self, text: str) -> np.ndarray: ...


@runtime_checkable
class ExporterProtocol(Protocol):
    """Protocol for corpus export formats."""

    def export(self, db: object, rubric_version_id: int, output_path: str) -> str: ...


@runtime_checkable
class NotificationProtocol(Protocol):
    """Protocol for user notifications."""

    def show(self, message: str, duration: int = 3000) -> None: ...

    def show_error(self, message: str, duration: int = 8000) -> None: ...

    def show_success(self, message: str, duration: int = 3000) -> None: ...

    def show_warning(self, message: str, duration: int = 5000) -> None: ...
