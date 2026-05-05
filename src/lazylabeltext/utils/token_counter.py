"""Token counting utilities.

Uses tiktoken's `cl100k_base` encoding when available. Falls back to a
word-length heuristic if tiktoken isn't installed OR can't load its
encoding file (e.g. first run without internet and no pre-cached
encoding bytes). The heuristic is rough (~4 chars/token) but accurate
enough for chunking decisions — token caps are heuristic anyway.

To work fully offline:
  1. On a machine with internet, run:
       python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
     This caches the encoding (~1.4 MB) in:
       Linux/Mac:  ~/.cache/tiktoken/
       Windows:    %LOCALAPPDATA%\\tiktoken_cache\\  (or $TEMP\\data-gym-cache\\)
  2. Copy the cache directory to the offline machine, OR set the env var
     TIKTOKEN_CACHE_DIR=<your-path> to point at it.
The path resolution is whatever tiktoken itself uses; this tool just
calls `tiktoken.get_encoding`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("lazylabeltext")

# _ENCODER is one of:
#   None       — never tried
#   False      — tiktoken unavailable / encoding load failed (use heuristic)
#   <encoder>  — live tiktoken encoder
_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    try:
        import tiktoken
    except ImportError:
        _ENCODER = False
        return _ENCODER
    try:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        # Most common cause: offline first-run, encoding file not cached.
        logger.warning(
            "tiktoken couldn't load 'cl100k_base' (%s). Falling back to a "
            "word-length token estimate. Pre-cache the encoding on a machine "
            "with internet: "
            "python -c \"import tiktoken; tiktoken.get_encoding('cl100k_base')\"",
            e,
        )
        _ENCODER = False
    return _ENCODER


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken, falling back to a word-based estimate."""
    encoder = _get_encoder()
    if encoder and encoder is not False:
        try:
            return len(encoder.encode(text))
        except Exception:
            # Defensive: if a single encode call fails (rare), fall through.
            return estimate_tokens(text)
    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """Fast heuristic token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def is_using_tiktoken() -> bool:
    """Whether token counts are exact (tiktoken) or heuristic. Useful in UI."""
    enc = _get_encoder()
    return bool(enc) and enc is not False
