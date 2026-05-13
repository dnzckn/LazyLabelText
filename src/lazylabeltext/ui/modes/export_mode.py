"""Export mode: format selection, preview, and export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.core.exporters import ExportFormat
from lazylabeltext.core.exporters.json_exporter import JSONExporter
from lazylabeltext.ui.modes.base_mode import BaseMode
from lazylabeltext.ui.workers.export_worker import ExportWorker

if TYPE_CHECKING:
    from lazylabeltext.core.app_context import AppContext


class ExportModeWidget(BaseMode):
    """Export labeled corpus to file."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self._worker: ExportWorker | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(10)

        header = QLabel("Export Labeled Corpus")
        header.setObjectName("sectionHeader")
        layout.addWidget(header)

        # Format selection
        layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItem("JSON", ExportFormat.JSON)
        layout.addWidget(self.format_combo)

        # Preview
        layout.addWidget(QLabel("Preview (manifest + first chunk):"))
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(300)
        self.preview.setPlaceholderText(
            "Click 'Preview' to render the manifest and first labeled chunk."
        )
        layout.addWidget(self.preview)

        # Buttons
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self._show_preview)
        layout.addWidget(self.preview_btn)

        self.export_btn = QPushButton("Export to File")
        self.export_btn.setObjectName("accentButton")
        self.export_btn.clicked.connect(self._export)
        layout.addWidget(self.export_btn)

        # Busy indicator. Indeterminate: the current exporter does the work
        # in one synchronous pass, so we have no per-chunk progress to wire.
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        self.progress.setFixedHeight(4)
        layout.addWidget(self.progress)

        # Status
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #888;")
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Preview — cheap, in-process, no parquet, no temp file.
    # ------------------------------------------------------------------

    def _show_preview(self) -> None:
        if self.ctx.rubric_manager is None or self.ctx.database is None:
            return

        rubric = self.ctx.rubric_manager.get_active_rubric()
        if rubric is None:
            self.preview.setPlainText("No rubric found.")
            return

        try:
            exporter = JSONExporter()
            data = exporter.build_preview(
                self.ctx.database, rubric.id, max_chunks=1
            )
            if not data["chunks"]:
                payload = {
                    "manifest": data["manifest"],
                    "rubric": data["rubric"],
                    "chunks": [],
                    "note": "No labeled chunks yet — label some first.",
                }
            else:
                payload = data
            text = json.dumps(payload, indent=2, ensure_ascii=False)
        except Exception as e:
            self.preview.setPlainText(f"Preview error: {e}")
            return

        self.preview.setPlainText(text)

    # ------------------------------------------------------------------
    # Export — async via QThread; UI stays responsive with a busy bar.
    # ------------------------------------------------------------------

    def _export(self) -> None:
        if self.ctx.rubric_manager is None or self.ctx.database is None:
            return

        rubric = self.ctx.rubric_manager.get_active_rubric()
        if rubric is None:
            return

        fmt = self.format_combo.currentData()
        ext_map = {ExportFormat.JSON: ("JSON (*.json)", "json")}
        filter_str, ext = ext_map.get(fmt, ("All (*)", "out"))

        # Default the dialog to the project's output directory — for folder
        # mode that's the corpus folder; for docmap mode it's the user-chosen
        # output dir where project.db lives. Falls back to home if neither is
        # set (no project loaded).
        s = self.ctx.settings
        default_dir = ""
        if s is not None:
            default_dir = (
                getattr(s, "last_project_output", "")
                or getattr(s, "last_project_path", "")
            )
        default_name = f"labeled_corpus.{ext}"
        default_path = (
            str(Path(default_dir) / default_name) if default_dir else default_name
        )

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Corpus", default_path, filter_str
        )
        if not path:
            return

        self._set_busy(True, f"Exporting to {Path(path).name}…")
        self._worker = ExportWorker(
            self.ctx.database, fmt, rubric.id, path, parent=self
        )
        self._worker.finished.connect(lambda p, f=fmt: self._on_export_done(p, f))
        self._worker.error.connect(self._on_export_error)
        self._worker.start()

    def _on_export_done(self, path: str, fmt: ExportFormat) -> None:
        self._set_busy(False)
        sidecar = Path(path).with_name(f"{Path(path).stem}_embeddings.parquet")
        suffix = f" (+ {sidecar.name})" if sidecar.exists() else ""
        self.status_label.setText(f"Exported to {path}{suffix}")
        self.status_label.setStyleSheet("color: #51cf66;")
        if self.ctx.audit_manager:
            self.ctx.audit_manager.log_event(
                "export_completed",
                payload={"format": fmt.value, "path": path},
            )
        self._worker = None

    def _on_export_error(self, msg: str) -> None:
        self._set_busy(False)
        self.status_label.setText(f"Export failed: {msg}")
        self.status_label.setStyleSheet("color: #ff6b6b;")
        self._worker = None

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.progress.setVisible(busy)
        self.export_btn.setEnabled(not busy)
        self.preview_btn.setEnabled(not busy)
        if busy:
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: #888;")
