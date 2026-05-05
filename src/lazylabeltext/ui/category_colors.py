"""Deterministic color assignment for rubric categories.

Used by the label timeline (cell fill) and the right-panel category cards
(left-edge stripe), so the same category gets the same color in both places.
The color depends only on the category's index in the rubric, not its name —
re-ordering categories will reshuffle colors, which is intentional (the
intent is "stable color across the same rubric version").
"""

from __future__ import annotations

from PyQt6.QtGui import QColor


def category_color(index: int) -> QColor:
    """Distinct hue per category, golden-angle spaced for separation."""
    hue = int((index * 137.508) % 360)
    return QColor.fromHsl(hue, 180, 130)


def color_map_for(categories) -> dict[str, QColor]:
    """Build a {category_name: QColor} map from a list of Category objects."""
    return {cat.name: category_color(i) for i, cat in enumerate(categories)}
