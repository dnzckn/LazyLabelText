"""OpenAI embedding provider (ada-002 and text-embedding-3-*)."""

from __future__ import annotations

import logging
import os

import numpy as np

from lazylabeltext.core.exceptions import EmbeddingProviderError

logger = logging.getLogger("lazylabeltext")


class OpenAIEmbeddingProvider:
    """Embedding provider using OpenAI's hosted embedding models.

    Supports text-embedding-ada-002 (1536-dim, legacy default),
    text-embedding-3-small (1536-dim, modern cheap), and
    text-embedding-3-large (3072-dim, modern best).
    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENAI_KEY")
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise EmbeddingProviderError(
                    "openai-embeddings", "openai package not installed"
                ) from e
            if not self.api_key:
                raise EmbeddingProviderError(
                    "openai-embeddings",
                    "No API key. Set OPENAI_API_KEY or configure in settings.",
                )
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def encode(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        try:
            response = client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
        except Exception as e:
            raise EmbeddingProviderError("openai-embeddings", str(e)) from e
        return np.array([d.embedding for d in response.data], dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def is_loaded(self) -> bool:
        return self._client is not None
