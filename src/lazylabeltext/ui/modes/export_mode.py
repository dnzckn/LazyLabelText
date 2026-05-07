"""Export mode: format selection, preview, and export."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.core.exporters import ExportFormat, export_corpus
from lazylabeltext.ui.modes.base_mode import BaseMode

if TYPE_CHECKING:
    from lazylabeltext.core.app_context import AppContext


class ExportModeWidget(BaseMode):
    """Export labeled corpus to file."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
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
        layout.addWidget(QLabel("Preview:"))
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(300)
        self.preview.setPlaceholderText("Click 'Preview' to see export output.")
        layout.addWidget(self.preview)

        # Buttons
        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self._show_preview)
        layout.addWidget(preview_btn)

        self.export_btn = QPushButton("Export to File")
        self.export_btn.setObjectName("accentButton")
        self.export_btn.clicked.connect(self._export)
        layout.addWidget(self.export_btn)

        # Status
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def _show_preview(self) -> None:
        if self.ctx.rubric_manager is None or self.ctx.database is None:
            return

        rubric = self.ctx.rubric_manager.get_active_rubric()
        if rubric is None:
            self.preview.setPlainText("No rubric found.")
            return

        import tempfile

        fmt = self.format_combo.currentData()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name

        try:
            export_corpus(fmt, self.ctx.database, rubric.id, tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                content = f.read()
            # Show first 5000 chars
            if len(content) > 5000:
                content = content[:5000] + "\n\n... (truncated)"
            self.preview.setPlainText(content)
        except Exception as e:
            self.preview.setPlainText(f"Preview error: {e}")

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

        try:
            export_corpus(fmt, self.ctx.database, rubric.id, path)
            self.status_label.setText(f"Exported to {path}")
            self.status_label.setStyleSheet("color: #51cf66;")

            if self.ctx.audit_manager:
                self.ctx.audit_manager.log_event(
                    "export_completed",
                    payload={"format": fmt.value, "path": path},
                )
        except Exception as e:
            self.status_label.setText(f"Export failed: {e}")
            self.status_label.setStyleSheet("color: #ff6b6b;")
