"""Embedding provider protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    """Protocol for text embedding providers."""

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def encode_one(self, text: str) -> np.ndarray: ...
