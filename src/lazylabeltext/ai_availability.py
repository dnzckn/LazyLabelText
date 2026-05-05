"""Detect availability of optional AI dependencies.

Uses importlib.util.find_spec so checking availability does NOT trigger
the expensive cold imports of torch / transformers / langchain. On
Windows in particular, importing sentence_transformers can take 15–30s
on first run because of DLL loading; doing that during MainWindow
construction looks like the app is hung. Lazy detection costs ~ms.
"""

from __future__ import annotations

import importlib.util


def _has(pkg: str) -> bool:
    """Return True iff `pkg` is importable, without actually importing it."""
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, ValueError):
        return False


# Any path to LLM classification — at least one of these must be installed
# for the labeling pipeline to work.
LLM_AVAILABLE = any(
    _has(p)
    for p in (
        "anthropic",
        "openai",
        "google.genai",  # google-genai exposes itself as `google.genai`
        "langchain_openai",
    )
)

# Any path to embeddings — sentence-transformers (local), OpenAI (hosted),
# or langchain_openai (Azure).
EMBEDDING_AVAILABLE = any(
    _has(p)
    for p in (
        "sentence_transformers",
        "openai",
        "langchain_openai",
    )
)

AI_AVAILABLE = LLM_AVAILABLE or EMBEDDING_AVAILABLE

LLM_INSTALL_HINT = "pip install lazylabeltext[include-ai]"
EMBEDDING_INSTALL_HINT = "pip install lazylabeltext[include-ai]"
