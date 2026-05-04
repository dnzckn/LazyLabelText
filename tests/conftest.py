"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Force offscreen rendering for CI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def tmp_project_dir():
    """Create a temporary directory for a project."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_database(tmp_project_dir):
    """Create a Database instance with the schema initialized."""
    from lazylabeltext.core.database import Database

    db_path = Path(tmp_project_dir) / "test.db"
    db = Database(str(db_path))
    yield db
    db.close()


@pytest.fixture
def sample_rubric_json():
    """Return a valid rubric JSON string."""
    return """{
        "categories": [
            {
                "name": "procedure",
                "definition": "Step-by-step instructions for performing a task.",
                "exemplars": ["First, open the valve.", "To configure the system..."],
                "confidence_threshold": 0.85
            },
            {
                "name": "principle",
                "definition": "A general rule or guideline.",
                "exemplars": ["Always verify before proceeding."],
                "confidence_threshold": 0.80
            }
        ]
    }"""


@pytest.fixture
def sample_document():
    """Return a ConvertedDocument instance."""
    from lazylabeltext.core.models import ConvertedDocument, Heading

    return ConvertedDocument(
        filename="test.md",
        format="md",
        full_text="# Introduction\n\nThis is the introduction.\n\n## Details\n\nSome details here.",
        headings=[
            Heading(level=1, text="Introduction", char_start=0, char_end=14),
            Heading(level=2, text="Details", char_start=42, char_end=52),
        ],
        status="parsed",
    )


def pytest_sessionfinish(session, exitstatus):
    """Clean up PyQt6 resources."""
    try:
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app:
            app.quit()
    except Exception:
        pass
