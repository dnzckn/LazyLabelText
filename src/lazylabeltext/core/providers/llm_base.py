"""LLM provider protocol."""

from __future__ import annotations

from typing import Protocol

from lazylabeltext.core.models import Category, ClassificationResult


class LLMProvider(Protocol):
    """Protocol for LLM classification providers."""

    def classify(
        self, chunk_text: str, categories: list[Category]
    ) -> ClassificationResult: ...
