"""OpenAI Chat Completions LLM provider."""

from __future__ import annotations

import logging
import os

from lazylabeltext.core.exceptions import LLMProviderError
from lazylabeltext.core.models import Category, ClassificationResult
from lazylabeltext.core.providers._classification import (
    build_classification_system_prompt,
    parse_classification_response,
)

logger = logging.getLogger("lazylabeltext")


class OpenAIProvider:
    """LLM classification using OpenAI's Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENAI_KEY")
        )
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise LLMProviderError(
                    "openai", "openai package not installed"
                ) from e

            if not self.api_key:
                raise LLMProviderError(
                    "openai",
                    "No API key. Set OPENAI_API_KEY or configure in settings.",
                )
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def complete(self, prompt: str, max_tokens: int = 4096) -> str:
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            raise LLMProviderError("openai", str(e)) from e
        return response.choices[0].message.content or ""

    def classify(
        self, chunk_text: str, categories: list[Category]
    ) -> ClassificationResult:
        client = self._get_client()
        system_prompt = build_classification_system_prompt(categories)
        user_message = f"Classify this text chunk:\n\n{chunk_text}"

        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
        except Exception as e:
            raise LLMProviderError("openai", str(e)) from e

        return parse_classification_response(
            response.choices[0].message.content or "", categories
        )

    def test_connection(self) -> tuple[bool, str]:
        try:
            client = self._get_client()
            client.chat.completions.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            return True, f"Connected to {self.model}"
        except Exception as e:
            return False, str(e)
