"""Exporter registry."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from lazylabeltext.core.database import Database


class ExportFormat(Enum):
    JSON = "JSON"
    # Phase 2: JSONL = "JSONL", PARQUET = "Parquet", CSV = "CSV"


class Exporter(Protocol):
    """Protocol for corpus exporters."""

    def export(self, db: Database, rubric_version_id: int, output_path: str) -> str: ...


EXPORTERS: dict[ExportFormat, Exporter] = {}


def _register(fmt: ExportFormat, exporter: Exporter) -> None:
    """Register an exporter for the given format."""
    EXPORTERS[fmt] = exporter


def export_corpus(
    fmt: ExportFormat,
    db: Database,
    rubric_version_id: int,
    output_path: str,
) -> str:
    """Export corpus using the specified format."""
    from lazylabeltext.core.exceptions import ExportFormatError

    exporter = EXPORTERS.get(fmt)
    if exporter is None:
        raise ExportFormatError(fmt.value, "No exporter registered")
    return exporter.export(db, rubric_version_id, output_path)


def available_formats() -> list[ExportFormat]:
    return list(EXPORTERS.keys())


# Import submodules to trigger registration
from lazylabeltext.core.exporters import json_exporter  # noqa: E402, F401
