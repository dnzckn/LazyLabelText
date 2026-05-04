"""Exception hierarchy for LazyLabelText."""

from __future__ import annotations


class LazyLabelTextError(Exception):
    """Base exception for all LazyLabelText errors."""


# --- Document errors ---


class DocumentError(LazyLabelTextError):
    """Base for document-related errors."""


class DocumentNotFoundError(DocumentError):
    def __init__(self, document_id: int) -> None:
        self.document_id = document_id
        super().__init__(f"Document not found: {document_id}")


class DocumentParseError(DocumentError):
    def __init__(self, file_path: str, reason: str) -> None:
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Failed to parse {file_path}: {reason}")


class UnsupportedFormatError(DocumentError):
    def __init__(self, file_path: str, fmt: str) -> None:
        self.file_path = file_path
        self.format = fmt
        super().__init__(f"Unsupported format '{fmt}' for {file_path}")


# --- Rubric errors ---


class RubricError(LazyLabelTextError):
    """Base for rubric-related errors."""


class RubricValidationError(RubricError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Rubric validation failed: {reason}")


class RubricVersionError(RubricError):
    def __init__(self, version_id: int, reason: str) -> None:
        self.version_id = version_id
        self.reason = reason
        super().__init__(f"Rubric version {version_id}: {reason}")


# --- Chunk errors ---


class ChunkError(LazyLabelTextError):
    """Base for chunk-related errors."""


class ChunkNotFoundError(ChunkError):
    def __init__(self, chunk_id: int) -> None:
        self.chunk_id = chunk_id
        super().__init__(f"Chunk not found: {chunk_id}")


class ChunkOperationError(ChunkError):
    def __init__(self, operation: str, reason: str) -> None:
        self.operation = operation
        self.reason = reason
        super().__init__(f"Chunk {operation} failed: {reason}")


# --- Label errors ---


class LabelError(LazyLabelTextError):
    """Base for label-related errors."""


class ClassificationError(LabelError):
    def __init__(self, chunk_id: int, reason: str) -> None:
        self.chunk_id = chunk_id
        self.reason = reason
        super().__init__(f"Classification failed for chunk {chunk_id}: {reason}")


class ReviewError(LabelError):
    def __init__(self, label_id: int, reason: str) -> None:
        self.label_id = label_id
        self.reason = reason
        super().__init__(f"Review failed for label {label_id}: {reason}")


# --- Provider errors ---


class ProviderError(LazyLabelTextError):
    """Base for provider-related errors."""


class LLMProviderError(ProviderError):
    def __init__(self, provider_name: str, reason: str) -> None:
        self.provider_name = provider_name
        self.reason = reason
        super().__init__(f"LLM provider '{provider_name}': {reason}")


class EmbeddingProviderError(ProviderError):
    def __init__(self, provider_name: str, reason: str) -> None:
        self.provider_name = provider_name
        self.reason = reason
        super().__init__(f"Embedding provider '{provider_name}': {reason}")


class ProviderNotConfiguredError(ProviderError):
    def __init__(self, provider_type: str) -> None:
        self.provider_type = provider_type
        super().__init__(f"No {provider_type} provider configured")


# --- Export errors ---


class ExportError(LazyLabelTextError):
    """Base for export-related errors."""


class ExportFormatError(ExportError):
    def __init__(self, format_name: str, reason: str) -> None:
        self.format_name = format_name
        self.reason = reason
        super().__init__(f"Export format '{format_name}': {reason}")


# --- Database errors ---


class DatabaseError(LazyLabelTextError):
    """Base for database-related errors."""


class DatabaseConnectionError(DatabaseError):
    def __init__(self, db_path: str, reason: str) -> None:
        self.db_path = db_path
        self.reason = reason
        super().__init__(f"Database connection failed ({db_path}): {reason}")


# --- Worker errors ---


class WorkerError(LazyLabelTextError):
    """Base for worker thread errors."""


class WorkerCancellationError(WorkerError):
    def __init__(self, worker_name: str) -> None:
        self.worker_name = worker_name
        super().__init__(f"Worker '{worker_name}' was cancelled")
