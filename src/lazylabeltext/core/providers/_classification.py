"""Shared classification prompt + response parsing.

All LLM providers (Anthropic, OpenAI, Google, Ollama) speak the same JSON
shape for `classify`, so the prompt construction and response parsing are
factored here.
"""

from __future__ import annotations

import json
import logging
import re

from lazylabeltext.core.models import Category, ClassificationResult

logger = logging.getLogger("lazylabeltext")


def build_classification_system_prompt(categories: list[Category]) -> str:
    """Render the rubric into a system prompt the LLM can classify against."""
    cat_descriptions: list[str] = []
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


def parse_classification_response(
    response_text: str, categories: list[Category]
) -> ClassificationResult:
    """Parse a JSON response (possibly fenced) into a ClassificationResult."""
    text = response_text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            logger.warning("Could not parse LLM classification response: %s", text[:200])
            return _dnb_result("Failed to parse LLM response")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return _dnb_result("Failed to parse LLM response")

    cats = data.get("predicted_categories") or []
    confs = data.get("confidence_per_category") or {}
    rationale = data.get("rationale", "")
    if not cats:
        # LLM made an explicit "no match" call — record it as DNB so the
        # timeline shows it differently from a never-labeled chunk.
        return _dnb_result(rationale or "No category match")
    return ClassificationResult(
        categories=cats,
        confidence_per_category=confs,
        rationale=rationale,
    )


def _dnb_result(rationale: str) -> ClassificationResult:
    """Build a ClassificationResult representing 'does not belong'."""
    from lazylabeltext.core.models import DNB_CATEGORY
    return ClassificationResult(
        categories=[DNB_CATEGORY],
        confidence_per_category={DNB_CATEGORY: 1.0},
        rationale=rationale,
    )
