"""Tests for text utility functions."""

from lazylabeltext.utils.text_utils import (
    compute_text_hash,
    normalize_whitespace,
    truncate_text,
)


def test_normalize_whitespace():
    assert normalize_whitespace("  hello   world  ") == "hello world"
    assert normalize_whitespace("a\n\nb\tc") == "a b c"


def test_truncate_text():
    assert truncate_text("short", 10) == "short"
    assert truncate_text("longer text here", 10) == "longer ..."
    assert len(truncate_text("x" * 300, 200)) == 200


def test_compute_text_hash():
    h1 = compute_text_hash("hello")
    h2 = compute_text_hash("hello")
    h3 = compute_text_hash("world")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # SHA-256 hex
