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
    try:
        if provider_type == "anthropic":
            from lazylabeltext.core.providers.anthropic_provider import (
                AnthropicProvider,
            )
            return AnthropicProvider(**kwargs)  # type: ignore[arg-type]
        if provider_type == "openai":
            from lazylabeltext.core.providers.openai_provider import OpenAIProvider
            return OpenAIProvider(**kwargs)  # type: ignore[arg-type]
        if provider_type == "google":
            from lazylabeltext.core.providers.google_provider import GoogleProvider
            return GoogleProvider(**kwargs)  # type: ignore[arg-type]
        if provider_type == "ollama":
            from lazylabeltext.core.providers.ollama_provider import OllamaProvider
            return OllamaProvider(**kwargs)  # type: ignore[arg-type]
        if provider_type in ("azure", "azure-openai"):
            from lazylabeltext.core.providers.azure_openai_provider import (
                AzureOpenAIProvider,
            )
            return AzureOpenAIProvider(**kwargs)  # type: ignore[arg-type]
    except Exception:
        return None
    return None


def create_embedding_provider(
    provider_type: str, **kwargs: object
) -> EmbeddingProviderProtocol | None:
    """Create an embedding provider by type name. 'none' / unknown → None."""
    if provider_type in ("none", "disabled", "off", ""):
        return None
    try:
        if provider_type == "sentence-transformers":
            from lazylabeltext.core.providers.sentence_transformer_provider import (
                SentenceTransformerProvider,
            )
            return SentenceTransformerProvider(**kwargs)  # type: ignore[arg-type]
        if provider_type in ("openai", "openai-embeddings"):
            from lazylabeltext.core.providers.openai_embedding_provider import (
                OpenAIEmbeddingProvider,
            )
            return OpenAIEmbeddingProvider(**kwargs)  # type: ignore[arg-type]
        if provider_type in ("azure", "azure-embeddings"):
            from lazylabeltext.core.providers.azure_embedding_provider import (
                AzureEmbeddingProvider,
            )
            return AzureEmbeddingProvider(**kwargs)  # type: ignore[arg-type]
    except Exception:
        return None
    return None
