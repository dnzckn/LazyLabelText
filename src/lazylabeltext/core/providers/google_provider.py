"""Google Gemini LLM provider (google-genai SDK)."""

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


class GoogleProvider:
    """LLM classification using Google's Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise LLMProviderError(
                    "google", "google-genai package not installed"
                ) from e

            if not self.api_key:
                raise LLMProviderError(
                    "google",
                    "No API key. Set GOOGLE_API_KEY or configure in settings.",
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def complete(self, prompt: str, max_tokens: int = 4096) -> str:
        client = self._get_client()
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=max_tokens),
            )
        except Exception as e:
            raise LLMProviderError("google", str(e)) from e
        return response.text or ""

    def classify(
        self, chunk_text: str, categories: list[Category]
    ) -> ClassificationResult:
        client = self._get_client()
        system_prompt = build_classification_system_prompt(categories)
        user_message = f"Classify this text chunk:\n\n{chunk_text}"

        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1024,
                    response_logprobs=True,
                ),
            )
        except Exception as e:
            # Some Gemini models reject response_logprobs; retry without.
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=1024,
                    ),
                )
            except Exception as e2:
                raise LLMProviderError("google", str(e2)) from e2

        result = parse_classification_response(response.text or "", categories)
        # Best-effort logprob extraction; never blocks the result.
        try:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                lp_result = getattr(candidates[0], "logprobs_result", None)
                if lp_result is not None:
                    chosen = getattr(lp_result, "chosen_candidates", None) or []
                    lp_values = [
                        c.log_probability
                        for c in chosen
                        if getattr(c, "log_probability", None) is not None
                    ]
                    if lp_values:
                        result.avg_logprob = sum(lp_values) / len(lp_values)
        except Exception:
            pass
        return result

    def test_connection(self) -> tuple[bool, str]:
        try:
            client = self._get_client()
            from google.genai import types

            client.models.generate_content(
                model=self.model,
                contents="Say OK",
                config=types.GenerateContentConfig(max_output_tokens=10),
            )
            return True, f"Connected to {self.model}"
        except Exception as e:
            return False, str(e)
