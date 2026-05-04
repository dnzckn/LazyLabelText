"""Sentence-transformers embedding provider."""

from __future__ import annotations

import logging

import numpy as np

from lazylabeltext.core.exceptions import EmbeddingProviderError

logger = logging.getLogger("lazylabeltext")


class SentenceTransformerProvider:
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise EmbeddingProviderError(
                    "sentence-transformers",
                    "sentence-transformers package not installed",
                ) from e

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into embeddings."""
        model = self._get_model()
        return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding."""
        result = self.encode([text])
        return result[0]

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
