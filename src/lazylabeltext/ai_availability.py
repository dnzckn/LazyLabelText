"""Detect availability of optional AI dependencies."""

from __future__ import annotations

LLM_AVAILABLE = False
EMBEDDING_AVAILABLE = False

try:
    import anthropic  # noqa: F401

    LLM_AVAILABLE = True
except ImportError:
    pass

try:
    import sentence_transformers  # noqa: F401

    EMBEDDING_AVAILABLE = True
except ImportError:
    pass

AI_AVAILABLE = LLM_AVAILABLE or EMBEDDING_AVAILABLE

LLM_INSTALL_HINT = "pip install lazylabeltext[include-ai]"
EMBEDDING_INSTALL_HINT = "pip install lazylabeltext[include-ai]"
