"""Token counting utilities.

Uses tiktoken's `cl100k_base` encoding when available. Falls back to a
word-length heuristic if tiktoken isn't installed OR can't load its
encoding file. The heuristic is rough (~4 chars/token) but accurate
enough for chunking decisions — token caps are heuristic anyway.

If tiktoken can't fetch its encoder file at runtime, you can drop it in
manually. Download:
    https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
and save (keeping the filename) to:
    <package_dir>/models/tiktoken/cl100k_base.tiktoken
``bootstrap_tiktoken_cache()`` (called from main.py on startup) detects
the drop, links it to the hash filename tiktoken expects, and points
``TIKTOKEN_CACHE_DIR`` at it — no further config needed.
"""

from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger("lazylabeltext")

# Canonical URL tiktoken downloads cl100k_base from. Used to derive the
# hash-named cache file tiktoken expects when looking inside
# TIKTOKEN_CACHE_DIR. Don't hardcode the hash — derive it so a future
# tiktoken release that changes the URL pattern stays correct.
_CL100K_BLOB_URL = (
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
)

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


def warm_encoder_cache() -> bool:
    """Eagerly load the tiktoken encoding on startup.

    Triggers tiktoken to load (and if needed, download + cache)
    `cl100k_base.tiktoken`. Later runs reuse the cache; if the load
    fails, the fallback in count_tokens() takes over.

    Returns True if tiktoken is now active.
    """
    return is_using_tiktoken()


def bootstrap_tiktoken_cache() -> bool:
    """Wire up a manually-downloaded BPE file to tiktoken's cache.

    Lookup path: ``<paths.models_dir>/tiktoken/cl100k_base.tiktoken``.
    If present, this function:
      1. Computes the SHA1 hash tiktoken uses as its cache key for the
         canonical cl100k_base URL.
      2. Copies (or links) the human-readable file to that hash name in
         the same directory, so tiktoken finds it without a download.
      3. Sets ``TIKTOKEN_CACHE_DIR`` to that directory if not already set.

    Returns True if the bootstrap completed (file found + linked + env
    var set), False if there was nothing to do or the link failed.
    """
    try:
        from lazylabeltext.config.paths import Paths
    except Exception:
        return False

    drop_dir = Paths().models_dir / "tiktoken"
    drop_file = drop_dir / "cl100k_base.tiktoken"
    if not drop_file.is_file():
        return False

    cache_key = hashlib.sha1(_CL100K_BLOB_URL.encode("utf-8")).hexdigest()
    target = drop_dir / cache_key
    if not target.exists():
        try:
            # Hardlink first (cheap, no copy); fall back to a byte copy
            # if hardlinks aren't allowed (cross-device, FAT32, etc.).
            try:
                os.link(drop_file, target)
            except OSError:
                import shutil
                shutil.copy2(drop_file, target)
        except OSError as e:
            logger.warning(
                "tiktoken bootstrap: couldn't link %s → %s (%s)",
                drop_file, target, e,
            )
            return False

    # Don't override an explicit user-set TIKTOKEN_CACHE_DIR — they may
    # be pointing at a shared cache deliberately.
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(drop_dir))
    logger.info("tiktoken cache bootstrapped from %s", drop_file)
    return True
