"""Propagation mode: batch apply chunking + labeling across corpus."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.ui.modes.base_mode import BaseMode
from lazylabeltext.ui.workers.propagation_worker import PropagationWorker

if TYPE_CHECKING:
    from lazylabeltext.core.app_context import AppContext
    from lazylabeltext.ui.main_window import MainWindow

logger = logging.getLogger("lazylabeltext")


class PropagationModeWidget(BaseMode):
    """Batch propagate chunking + labeling across all documents."""

    def __init__(
        self,
        context: AppContext,
        main_window: MainWindow,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(context, parent)
        self.main_window = main_window
        self._worker: PropagationWorker | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        # Header
        header = QLabel("Propagate Across Corpus")
        header.setObjectName("sectionHeader")
        layout.addWidget(header)

        info = QLabel(
            "Apply the current chunking parameters and rubric to all documents. "
            "Fine-tune on a single document first, then propagate."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888;")
        layout.addWidget(info)

        # Document status table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Document", "Status", "Chunks", "Labels"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.table, 1)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("color: #888;")
        layout.addWidget(self.progress_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.propagate_btn = QPushButton("Propagate All")
        self.propagate_btn.setObjectName("accentButton")
        self.propagate_btn.clicked.connect(self._start_propagation)
        btn_layout.addWidget(self.propagate_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("dangerButton")
        self.cancel_btn.clicked.connect(self._cancel_propagation)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def activate(self) -> None:
        self._refresh_table()

    def _refresh_table(self) -> None:
        if self.ctx.database is None:
            return

        status = self.ctx.database.get_document_label_status()
        self.table.setRowCount(len(status))

        for row, s in enumerate(status):
            self.table.setItem(row, 0, QTableWidgetItem(s["filename"]))
            self.table.setItem(row, 1, QTableWidgetItem(s["status"]))
            self.table.setItem(row, 2, QTableWidgetItem(str(s["chunk_count"])))
            self.table.setItem(row, 3, QTableWidgetItem(str(s["label_count"])))

    def _start_propagation(self) -> None:
        if self.ctx.propagation_manager is None or self.ctx.rubric_manager is None:
            return

        rubric = self.ctx.rubric_manager.get_active_rubric()
        if rubric is None:
            self.main_window.notification_manager.show_warning("Create a rubric first")
            return

        self.propagate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(0)

        # Get current chunking params from settings
        settings = self.ctx.settings
        params = {
            "min_tokens": settings.chunk_min_tokens if settings else 50,
            "max_tokens": settings.chunk_max_tokens if settings else 800,
            "heading_split_levels": settings.heading_split_levels
            if settings
            else [1, 2, 3],
        }
        strategy = settings.default_chunk_strategy if settings else "structural"

        self._worker = PropagationWorker(
            self.ctx.propagation_manager, strategy, params, rubric
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_result.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _cancel_propagation(self) -> None:
        if self._worker:
            self._worker.cancel()

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(message)

    def _on_finished(self, results: dict) -> None:
        self.propagate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        self.progress_label.setText(
            f"Done: {results['processed']}/{results['total']} docs, "
            f"{results['chunks_created']} chunks, {results['labels_created']} labels"
        )

        if results.get("errors"):
            self.main_window.notification_manager.show_warning(
                f"Completed with {len(results['errors'])} errors"
            )
        else:
            self.main_window.notification_manager.show_success("Propagation complete")

        self._refresh_table()
        self.main_window._update_stats()

    def _on_error(self, msg: str) -> None:
        self.propagate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.main_window.notification_manager.show_error(f"Propagation failed: {msg}")
