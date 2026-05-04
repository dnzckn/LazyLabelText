"""Label mode: keyboard-driven single-chunk review."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.ui.modes.base_mode import BaseMode
from lazylabeltext.ui.widgets.confidence_bar import ConfidenceBar

if TYPE_CHECKING:
    from lazylabeltext.core.app_context import AppContext
    from lazylabeltext.core.models import Chunk, Label
    from lazylabeltext.ui.main_window import MainWindow

logger = logging.getLogger("lazylabeltext")


class LabelModeWidget(BaseMode):
    """Single-chunk review interface with keyboard shortcuts."""

    def __init__(
        self,
        context: AppContext,
        main_window: MainWindow,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(context, parent)
        self.main_window = main_window
        self._queue: list[tuple[Chunk, Label]] = []
        self._current_index = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(12)

        # Progress
        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self.progress_label)

        # Confidence bars
        self.confidence_container = QWidget()
        self.confidence_layout = QVBoxLayout(self.confidence_container)
        self.confidence_layout.setSpacing(4)
        layout.addWidget(self.confidence_container)

        # Chunk text
        self.chunk_text = QTextEdit()
        self.chunk_text.setReadOnly(True)
        self.chunk_text.setStyleSheet("font-size: 13px; padding: 10px;")
        layout.addWidget(self.chunk_text, 1)

        # Source citation
        self.source_label = QLabel()
        self.source_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.source_label)

        # Rationale
        self.rationale_label = QLabel()
        self.rationale_label.setWordWrap(True)
        self.rationale_label.setStyleSheet(
            "color: #aaa; font-style: italic; font-size: 11px;"
        )
        layout.addWidget(self.rationale_label)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        actions = [
            ("Accept (Space)", "accentButton", self.accept_current),
            ("Correct (C)", None, self._correct_current),
            ("Skip (S)", None, self._skip_current),
            ("Flag (F)", "dangerButton", self._flag_current),
        ]
        for text, obj_name, callback in actions:
            btn = QPushButton(text)
            if obj_name:
                btn.setObjectName(obj_name)
            btn.clicked.connect(callback)
            btn_layout.addWidget(btn)

        layout.addLayout(btn_layout)

        # Run labeling button (shown when no labels exist)
        self.run_label_btn = QPushButton("Run LLM Labeling on Current Document")
        self.run_label_btn.setObjectName("accentButton")
        self.run_label_btn.clicked.connect(self._run_labeling)
        layout.addWidget(self.run_label_btn)
        self.run_label_btn.hide()

    def activate(self) -> None:
        self._load_review_queue()

    def _load_review_queue(self) -> None:
        if self.ctx.label_manager is None or self.ctx.rubric_manager is None:
            self.progress_label.setText("Open a project first.")
            return

        rubric = self.ctx.rubric_manager.get_active_rubric()
        if rubric is None:
            self.progress_label.setText("Create a rubric first.")
            return

        # Get all labeled chunks for review (not just low-confidence)
        self._queue = self.ctx.label_manager.get_review_queue(rubric.id, threshold=1.0)

        if not self._queue:
            # Check if there are unlabeled chunks
            unlabeled = self.ctx.label_manager.get_unlabeled_chunks(rubric.id)
            if unlabeled:
                self.progress_label.setText(f"{len(unlabeled)} chunks need labeling.")
                self.run_label_btn.show()
            else:
                self.progress_label.setText("All chunks reviewed!")
                self.run_label_btn.hide()
            return

        self.run_label_btn.hide()
        self._current_index = 0
        self._show_current()

    def _show_current(self) -> None:
        if not self._queue or self._current_index >= len(self._queue):
            self.progress_label.setText("Review complete!")
            self.chunk_text.clear()
            self.rationale_label.clear()
            self.source_label.clear()
            return

        chunk, label = self._queue[self._current_index]

        self.progress_label.setText(
            f"Reviewing {self._current_index + 1} of {len(self._queue)} "
            f"(confidence: {label.composite_confidence:.0%})"
        )

        self.chunk_text.setPlainText(chunk.text)

        # Source
        section = " > ".join(chunk.section_path) if chunk.section_path else ""
        self.source_label.setText(
            f"Source: chars {chunk.char_start}-{chunk.char_end} | "
            f"{chunk.token_count} tokens | {section}"
        )

        # Rationale
        self.rationale_label.setText(f"LLM rationale: {label.rationale}")

        # Confidence bars
        while self.confidence_layout.count():
            child = self.confidence_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for cat, conf in sorted(
            label.confidence_per_category.items(), key=lambda x: -x[1]
        ):
            bar = ConfidenceBar(value=conf, label=cat)
            self.confidence_layout.addWidget(bar)

    def accept_current(self) -> None:
        self._submit_review("accept")

    def _correct_current(self) -> None:
        self._submit_review("correct")

    def _skip_current(self) -> None:
        self._submit_review("skip")

    def _flag_current(self) -> None:
        self._submit_review("flag")

    def _submit_review(self, action: str) -> None:
        if not self._queue or self._current_index >= len(self._queue):
            return

        _, label = self._queue[self._current_index]

        if self.ctx.label_manager is None:
            return

        try:
            final_cats = label.predicted_categories if action == "accept" else []
            self.ctx.label_manager.submit_review(
                label.id, action, final_categories=final_cats
            )
        except Exception as e:
            logger.error("Review failed: %s", e)

        self._current_index += 1
        self._show_current()
        self.main_window._update_stats()

    def _run_labeling(self) -> None:
        """Run LLM labeling on all unlabeled chunks."""
        if self.ctx.label_manager is None or self.ctx.rubric_manager is None:
            return

        rubric = self.ctx.rubric_manager.get_active_rubric()
        if rubric is None:
            return

        unlabeled = self.ctx.label_manager.get_unlabeled_chunks(rubric.id)
        if not unlabeled:
            return

        self.run_label_btn.setEnabled(False)
        self.progress_label.setText(f"Labeling {len(unlabeled)} chunks...")

        try:
            self.ctx.label_manager.label_batch(
                unlabeled,
                rubric,
                progress_callback=lambda i, t: self.progress_label.setText(
                    f"Labeling chunk {i}/{t}..."
                ),
            )
            self.main_window.notification_manager.show_success(
                f"Labeled {len(unlabeled)} chunks"
            )
            self._load_review_queue()
        except Exception as e:
            self.main_window.notification_manager.show_error(f"Labeling failed: {e}")
        finally:
            self.run_label_btn.setEnabled(True)

    def handle_escape(self) -> None:
        pass
