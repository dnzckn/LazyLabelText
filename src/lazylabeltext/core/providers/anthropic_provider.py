"""Anthropic Claude LLM provider."""

from __future__ import annotations

import logging
import os
import threading

from lazylabeltext.core.exceptions import LLMProviderError
from lazylabeltext.core.models import Category, ClassificationResult
from lazylabeltext.core.providers._classification import (
    build_classification_system_prompt,
    parse_classification_response,
)

logger = logging.getLogger("lazylabeltext")


class AnthropicProvider:
    """LLM classification using Anthropic's Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_KEY")
        )
        self.model = model
        self._client = None
        self._client_lock = threading.Lock()

    def _get_client(self):
        # Double-checked locking: cheap read fast-path once the client exists,
        # serialized init the first time so concurrent threads don't both
        # construct the SDK client.
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                import anthropic
            except ImportError as e:
                raise LLMProviderError(
                    "anthropic", "anthropic package not installed"
                ) from e

            if not self.api_key:
                raise LLMProviderError(
                    "anthropic",
                    "No API key. Set ANTHROPIC_API_KEY or configure in settings.",
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(self, prompt: str, max_tokens: int = 4096) -> str:
        """Generic single-turn completion. Used by non-classification callers."""
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            raise LLMProviderError("anthropic", str(e)) from e
        return response.content[0].text

    def classify(
        self, chunk_text: str, categories: list[Category]
    ) -> ClassificationResult:
        """Classify a text chunk against rubric categories."""
        client = self._get_client()

        system_prompt = build_classification_system_prompt(categories)
        user_message = f"Classify this text chunk:\n\n{chunk_text}"

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as e:
            raise LLMProviderError("anthropic", str(e)) from e

        return parse_classification_response(response.content[0].text, categories)

    def test_connection(self) -> tuple[bool, str]:
        """Test the API connection. Returns (success, message)."""
        try:
            client = self._get_client()
            client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            return True, f"Connected to {self.model}"
        except Exception as e:
            return False, str(e)
