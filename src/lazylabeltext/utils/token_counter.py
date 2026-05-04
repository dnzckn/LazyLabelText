"""Token counting utilities."""

from __future__ import annotations

_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _ENCODER = False  # sentinel: tiktoken not available
    return _ENCODER


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken, falling back to word-based estimate."""
    encoder = _get_encoder()
    if encoder and encoder is not False:
        return len(encoder.encode(text))
    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """Fast heuristic token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)
