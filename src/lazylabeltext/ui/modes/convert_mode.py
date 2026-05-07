"""Convert mode: side-by-side original and converted text view."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.ui.modes.base_mode import BaseMode
from lazylabeltext.ui.workers.reconvert_worker import ReconvertWorker

if TYPE_CHECKING:
    from lazylabeltext.core.app_context import AppContext


class ConvertModeWidget(BaseMode):
    """Shows original document alongside converted structured text."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self._current_doc_id: int | None = None
        self._reconvert_worker: ReconvertWorker | None = None
        self._setup_ui()
        self._sync_toggles_from_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: original text
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 4, 4, 4)
        left_header = QLabel("Source Text")
        left_header.setObjectName("sectionHeader")
        left_layout.addWidget(left_header)
        self.original_view = QTextEdit()
        self.original_view.setReadOnly(True)
        self.original_view.setPlaceholderText(
            "Select a document from the left panel to view its contents."
        )
        left_layout.addWidget(self.original_view)
        splitter.addWidget(left)

        # Right: converted text with heading highlights
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 8, 4)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        right_header = QLabel("Converted (with structure)")
        right_header.setObjectName("sectionHeader")
        header_row.addWidget(right_header)
        header_row.addStretch(1)

        self.high_fidelity_check = QCheckBox("High-fidelity")
        self.high_fidelity_check.setToolTip(
            "Use docling for better tables and figures (slower; "
            "downloads ~1–2 GB of models on first run)."
        )
        self.high_fidelity_check.toggled.connect(self._on_high_fidelity_toggled)
        header_row.addWidget(self.high_fidelity_check)

        self.ocr_check = QCheckBox("OCR")
        self.ocr_check.setToolTip(
            "Run OCR for scanned PDFs. Requires high-fidelity mode."
        )
        self.ocr_check.toggled.connect(self._on_ocr_toggled)
        header_row.addWidget(self.ocr_check)

        self.reconvert_btn = QPushButton("Reconvert")
        self.reconvert_btn.setToolTip(
            "Re-run conversion on the selected document with the current settings."
        )
        self.reconvert_btn.setEnabled(False)
        self.reconvert_btn.clicked.connect(self._on_reconvert_clicked)
        header_row.addWidget(self.reconvert_btn)

        right_layout.addLayout(header_row)

        self.converted_view = QTextEdit()
        self.converted_view.setReadOnly(True)
        self.converted_view.setPlaceholderText("Converted text will appear here.")
        right_layout.addWidget(self.converted_view)

        # Info bar
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #888; font-size: 11px; padding: 2px;")
        right_layout.addWidget(self.info_label)
        splitter.addWidget(right)

        splitter.setSizes([400, 400])
        layout.addWidget(splitter)

    # --- Settings wiring -------------------------------------------------

    def _sync_toggles_from_settings(self) -> None:
        s = self.ctx.settings
        if s is None:
            self.ocr_check.setEnabled(False)
            return
        self.high_fidelity_check.blockSignals(True)
        self.ocr_check.blockSignals(True)
        self.high_fidelity_check.setChecked(
            getattr(s, "use_high_fidelity_conversion", False)
        )
        self.ocr_check.setChecked(getattr(s, "high_fidelity_ocr", False))
        self.ocr_check.setEnabled(self.high_fidelity_check.isChecked())
        self.high_fidelity_check.blockSignals(False)
        self.ocr_check.blockSignals(False)

    def _persist_settings(self) -> None:
        if self.ctx.settings is None or self.ctx.paths is None:
            return
        with contextlib.suppress(Exception):
            self.ctx.settings.save_to_file(str(self.ctx.paths.settings_file))

    def _on_high_fidelity_toggled(self, checked: bool) -> None:
        if self.ctx.settings is not None:
            self.ctx.settings.use_high_fidelity_conversion = checked
            if not checked:
                # OCR is meaningless without docling; force it off so the
                # persisted state stays consistent with what's selectable.
                self.ctx.settings.high_fidelity_ocr = False
                self.ocr_check.blockSignals(True)
                self.ocr_check.setChecked(False)
                self.ocr_check.blockSignals(False)
        self.ocr_check.setEnabled(checked)
        self._persist_settings()

    def _on_ocr_toggled(self, checked: bool) -> None:
        if self.ctx.settings is not None:
            self.ctx.settings.high_fidelity_ocr = checked
        self._persist_settings()

    # --- Reconvert -------------------------------------------------------

    def _on_reconvert_clicked(self) -> None:
        if (
            self._current_doc_id is None
            or self.ctx.document_manager is None
            or self._reconvert_worker is not None
        ):
            return

        worker = ReconvertWorker(
            self.ctx.document_manager, self._current_doc_id, parent=self
        )

        original_text = self.reconvert_btn.text()
        self.reconvert_btn.setEnabled(False)
        self.reconvert_btn.setText("Reconverting…")
        self.info_label.setText("Reconverting…")

        def _restore() -> None:
            self.reconvert_btn.setEnabled(self._current_doc_id is not None)
            self.reconvert_btn.setText(original_text)
            self._reconvert_worker = None

        def _on_status(msg: str) -> None:
            if msg:
                self.info_label.setText(msg)

        def _on_ok(new_id: int) -> None:
            _restore()
            self._current_doc_id = new_id
            self.on_document_selected(new_id)

        def _on_failed(err: str) -> None:
            _restore()
            QMessageBox.warning(self, "Reconvert failed", err)

        worker.status_changed.connect(_on_status)
        worker.finished_ok.connect(_on_ok)
        worker.failed.connect(_on_failed)

        self._reconvert_worker = worker
        worker.start()

    # --- Document display ------------------------------------------------

    def on_document_selected(self, doc_id: int) -> None:
        if self.ctx.document_manager is None:
            return

        doc = self.ctx.document_manager.get_document(doc_id)
        if doc is None:
            self._current_doc_id = None
            self.reconvert_btn.setEnabled(False)
            return

        self._current_doc_id = doc_id
        self.reconvert_btn.setEnabled(True)

        self.original_view.setPlainText(doc.full_text)

        # Build converted view with highlighted headings
        html_parts = []
        heading_colors = {
            1: "#5c8fbf",
            2: "#7faf5c",
            3: "#bf8f5c",
            4: "#8f5cbf",
            5: "#5cbfbf",
            6: "#bf5c5c",
        }

        text = doc.full_text
        last_end = 0

        for heading in sorted(doc.headings, key=lambda h: h.char_start):
            if heading.char_start > last_end:
                before = text[last_end : heading.char_start]
                html_parts.append(
                    f"<span style='color: #ccc;'>{self._escape(before)}</span>"
                )

            color = heading_colors.get(heading.level, "#aaa")
            heading_text = text[heading.char_start : heading.char_end]
            size = max(12, 20 - heading.level * 2)
            html_parts.append(
                f"<div style='color: {color}; font-size: {size}px; font-weight: bold; "
                f"margin: 4px 0;'>"
                f"{'#' * heading.level} {self._escape(heading_text)}</div>"
            )
            last_end = heading.char_end

        if last_end < len(text):
            html_parts.append(
                f"<span style='color: #ccc;'>{self._escape(text[last_end:])}</span>"
            )

        self.converted_view.setHtml(
            "<div style='white-space: pre-wrap; font-family: monospace;'>"
            + "".join(html_parts)
            + "</div>"
        )

        backend = doc.metadata.get("backend", "default")
        info = (
            f"{len(doc.full_text):,} chars | "
            f"{len(doc.headings)} headings | "
            f"{len(doc.pages)} pages | "
            f"Format: {doc.format} | "
            f"Backend: {backend}"
        )
        n_tables = len(doc.metadata.get("tables", []) or [])
        n_figures = len(doc.metadata.get("figures", []) or [])
        if n_tables:
            info += f" | {n_tables} tables"
        if n_figures:
            info += f" | {n_figures} figures"
        if doc.warnings:
            info += f" | Warnings: {len(doc.warnings)}"
        self.info_label.setText(info)

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
