"""Ollama local LLM provider (HTTP API)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from lazylabeltext.core.exceptions import LLMProviderError
from lazylabeltext.core.models import Category, ClassificationResult
from lazylabeltext.core.providers._classification import (
    build_classification_system_prompt,
    parse_classification_response,
)

logger = logging.getLogger("lazylabeltext")


class OllamaProvider:
    """LLM classification via a local Ollama server (no API key)."""

    def __init__(
        self,
        api_key: str | None = None,  # noqa: ARG002 — accepted for symmetry
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict, timeout: float = 120.0) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as e:
            raise LLMProviderError("ollama", f"HTTP error: {e}") from e
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise LLMProviderError(
                "ollama", f"Non-JSON response: {body[:200]}"
            ) from e

    def complete(self, prompt: str, max_tokens: int = 4096) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        data = self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    def classify(
        self, chunk_text: str, categories: list[Category]
    ) -> ClassificationResult:
        system_prompt = build_classification_system_prompt(categories)
        user_message = f"Classify this text chunk:\n\n{chunk_text}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "format": "json",
            "options": {"num_predict": 1024},
        }
        data = self._post("/api/chat", payload)
        text = data.get("message", {}).get("content", "")
        return parse_classification_response(text, categories)

    def test_connection(self) -> tuple[bool, str]:
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Say OK"}],
                "stream": False,
                "options": {"num_predict": 10},
            }
            self._post("/api/chat", payload, timeout=20.0)
            return True, f"Connected to {self.model} at {self.base_url}"
        except Exception as e:
            return False, str(e)
