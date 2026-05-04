"""Startup banner display for LazyLabelText."""

from __future__ import annotations

import random

from lazylabeltext import __version__

_BANNER = r"""
  _                    _         _          _ _____         _
 | |   __ _ _____   _ | |   __ _| |__   ___| |_   _|____  _| |_
 | |  / _` |_  / | | || |  / _` | '_ \ / _ \ | | |/ _ \ \/ / __|
 | |_| (_| |/ /| |_| || |_| (_| | |_) |  __/ | | |  __/>  <| |_
 |____\__,_/___|\__, ||____\__,_|_.__/ \___|_| |_|\___/_/\_\\__|
                |___/
"""

_TIPS = [
    "Press F1-F7 to switch modes: Convert, Rubric, Chunk, Label, Results, Propagate, Export.",
    "Space accepts the current label, S skips, F flags for review.",
    "Labels are sorted by confidence -- you see the hardest cases first.",
    "Use M to merge adjacent chunks, T to split at cursor.",
    "The rubric is versioned -- every edit creates a new version.",
    "Set ANTHROPIC_API_KEY to enable LLM labeling.",
    "Export includes a manifest with corpus statistics for reproducibility.",
    "All state changes are logged to the audit trail automatically.",
    "Adjust chunking parameters and see results update instantly.",
    "The rubric editor shows live label predictions as you edit categories.",
]


def show_banner() -> None:
    """Print the startup banner and a random tip."""
    print(f"\033[36m{_BANNER}\033[0m")
    print(f"  v{__version__}")
    print(f"  Tip: {random.choice(_TIPS)}")
    print()
