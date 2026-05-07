"""OpenAI Chat Completions LLM provider."""

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
        self._client_lock = threading.Lock()

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
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
                logprobs=True,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
        except Exception as e:
            raise LLMProviderError("openai", str(e)) from e

        result = parse_classification_response(
            response.choices[0].message.content or "", categories
        )
        # Best-effort logprob extraction; never blocks the result.
        try:
            content = response.choices[0].logprobs.content  # type: ignore[union-attr]
            if content:
                lp_values = [tok.logprob for tok in content if tok.logprob is not None]
                if lp_values:
                    result.avg_logprob = sum(lp_values) / len(lp_values)
        except Exception:
            pass
        return result

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
