"""Tests for settings persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

from lazylabeltext.config.settings import Settings


def test_default_settings():
    s = Settings()
    assert s.dark_mode is True
    assert s.chunk_min_tokens == 50
    assert s.auto_accept_threshold == 0.0


def test_save_and_load():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    s = Settings(window_width=1920, dark_mode=False)
    s.save_to_file(path)

    loaded = Settings.load_from_file(path)
    assert loaded.window_width == 1920
    assert loaded.dark_mode is False


def test_load_missing_file():
    s = Settings.load_from_file("/nonexistent/path.json")
    assert s.window_width == 1600  # default


def test_update():
    s = Settings()
    s.update(window_width=800, dark_mode=False)
    assert s.window_width == 800
    assert s.dark_mode is False


def test_api_key_not_persisted():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    s = Settings(llm_api_key="secret-key-123")
    s.save_to_file(path)

    content = Path(path).read_text()
    assert "secret-key-123" not in content
