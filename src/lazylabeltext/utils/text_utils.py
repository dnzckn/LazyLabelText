"""Text processing utility functions."""

from __future__ import annotations

import hashlib
import re


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to single spaces and strip."""
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, max_chars: int = 200, ellipsis: str = "...") -> str:
    """Truncate text to max_chars, adding ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(ellipsis)] + ellipsis


def extract_first_sentence(text: str) -> str:
    """Extract the first sentence from text."""
    match = re.match(r"^(.*?[.!?])\s", text)
    if match:
        return match.group(1)
    return text[:200] if len(text) > 200 else text


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 hash of text for cache keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
