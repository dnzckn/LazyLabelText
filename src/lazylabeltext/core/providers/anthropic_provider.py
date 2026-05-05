"""Anthropic Claude LLM provider."""

from __future__ import annotations

import json
import logging
import os

from lazylabeltext.core.exceptions import LLMProviderError
from lazylabeltext.core.models import Category, ClassificationResult

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

    def _get_client(self):
        if self._client is None:
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

        system_prompt = self._build_system_prompt(categories)
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

        return self._parse_response(response.content[0].text, categories)

    def _build_system_prompt(self, categories: list[Category]) -> str:
        cat_descriptions = []
        for cat in categories:
            desc = f"- **{cat.name}**: {cat.definition}"
            if cat.exemplars:
                examples = "; ".join(f'"{e}"' for e in cat.exemplars[:3])
                desc += f"\n  Examples: {examples}"
            if cat.boundary_cases:
                desc += f"\n  Boundary cases: {'; '.join(cat.boundary_cases[:2])}"
            cat_descriptions.append(desc)

        categories_text = "\n\n".join(cat_descriptions)

        return f"""You are a text classification assistant. Given a text chunk, classify it into one or more of the following categories.

CATEGORIES:
{categories_text}

INSTRUCTIONS:
- Analyze the chunk carefully against each category definition.
- Assign the most appropriate category or categories.
- Provide a confidence score (0.0 to 1.0) for each assigned category.
- Write a brief rationale explaining your classification.

Respond with valid JSON only, no other text:
{{
  "predicted_categories": ["category_name"],
  "confidence_per_category": {{"category_name": 0.95}},
  "rationale": "Brief explanation of why this classification was chosen."
}}"""

    def _parse_response(
        self, response_text: str, categories: list[Category]
    ) -> ClassificationResult:
        """Parse the LLM response into a ClassificationResult."""
        # Extract JSON from response (handle markdown code blocks)
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON: %s", text[:200])
            # Fallback: assign first category with low confidence
            if categories:
                return ClassificationResult(
                    categories=[categories[0].name],
                    confidence_per_category={categories[0].name: 0.1},
                    rationale="Failed to parse LLM response",
                )
            return ClassificationResult()

        return ClassificationResult(
            categories=data.get("predicted_categories", []),
            confidence_per_category=data.get("confidence_per_category", {}),
            rationale=data.get("rationale", ""),
        )

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
