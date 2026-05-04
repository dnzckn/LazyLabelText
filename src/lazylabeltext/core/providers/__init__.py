"""LLM and embedding provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lazylabeltext.core.protocols import (
        EmbeddingProviderProtocol,
        LLMProviderProtocol,
    )


def create_llm_provider(
    provider_type: str, **kwargs: object
) -> LLMProviderProtocol | None:
    """Create an LLM provider by type name."""
    if provider_type == "anthropic":
        try:
            from lazylabeltext.core.providers.anthropic_provider import (
                AnthropicProvider,
            )

            return AnthropicProvider(**kwargs)  # type: ignore[arg-type]
        except (ImportError, Exception):
            return None
    return None


def create_embedding_provider(
    provider_type: str, **kwargs: object
) -> EmbeddingProviderProtocol | None:
    """Create an embedding provider by type name."""
    if provider_type == "sentence-transformers":
        try:
            from lazylabeltext.core.providers.sentence_transformer_provider import (
                SentenceTransformerProvider,
            )

            return SentenceTransformerProvider(**kwargs)  # type: ignore[arg-type]
        except (ImportError, Exception):
            return None
    return None
