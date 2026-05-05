"""Label mode: timeline-based chunk review for the active document."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.ui.modes.base_mode import BaseMode
from lazylabeltext.ui.widgets.confidence_bar import ConfidenceBar
from lazylabeltext.ui.widgets.timeline_widget import ZoomableTimeline
from lazylabeltext.ui.workers.labeling_worker import LabelingWorker

if TYPE_CHECKING:
    from lazylabeltext.core.app_context import AppContext
    from lazylabeltext.core.models import Chunk, HumanReview, Label, Rubric
    from lazylabeltext.ui.main_window import MainWindow

logger = logging.getLogger("lazylabeltext")


def _category_color(index: int) -> QColor:
    """Deterministic distinct-hue color for a category at the given index."""
    hue = int((index * 137.508) % 360)
    return QColor.fromHsl(hue, 180, 130)


class LabelModeWidget(BaseMode):
    """Timeline navigator + per-chunk review for the active document."""

    def __init__(
        self,
        context: AppContext,
        main_window: MainWindow,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(context, parent)
        self.main_window = main_window

        self._chunks: list[Chunk] = []
        self._labels_by_chunk: dict[int, Label] = {}
        self._reviews_by_label: dict[int, HumanReview] = {}
        self._current_index = 0
        self._labeling_worker: LabelingWorker | None = None
        self._labeling_total = 0
        self._setup_ui()
        self._setup_shortcuts()

    # --- UI setup --------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(8)

        # Timeline at the top — visual map of all chunks
        self.timeline = ZoomableTimeline()
        self.timeline.frame_selected.connect(self._on_timeline_select)
        layout.addWidget(self.timeline)

        # Nav row: Prev/Next + position label
        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self._go_prev)
        nav_row.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self._go_next)
        nav_row.addWidget(self.next_btn)

        nav_row.addSpacing(12)

        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("color: #aaa; font-size: 12px;")
        nav_row.addWidget(self.progress_label)
        nav_row.addStretch()
        layout.addLayout(nav_row)

        # Confidence bars (per predicted category)
        self.confidence_container = QWidget()
        self.confidence_layout = QVBoxLayout(self.confidence_container)
        self.confidence_layout.setSpacing(4)
        self.confidence_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.confidence_container)

        # Chunk text
        self.chunk_text = QTextEdit()
        self.chunk_text.setReadOnly(True)
        self.chunk_text.setStyleSheet("font-size: 13px; padding: 10px;")
        layout.addWidget(self.chunk_text, 1)

        # Source citation
        self.source_label = QLabel()
        self.source_label.setStyleSheet("color: #888; font-size: 11px;")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

        # Rationale
        self.rationale_label = QLabel()
        self.rationale_label.setWordWrap(True)
        self.rationale_label.setStyleSheet(
            "color: #aaa; font-style: italic; font-size: 11px;"
        )
        layout.addWidget(self.rationale_label)

        # Action buttons (operate on the current chunk)
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        actions = [
            ("Accept (Space)", "accentButton", self.accept_current),
            ("Correct (C)", None, self._correct_current),
            ("Skip (S)", None, self._skip_current),
            ("Flag (F)", "dangerButton", self._flag_current),
            ("Discard (D)", "dangerButton", self._discard_current),
        ]
        for text, obj_name, callback in actions:
            btn = QPushButton(text)
            if obj_name:
                btn.setObjectName(obj_name)
            btn.clicked.connect(callback)
            action_row.addWidget(btn)
        layout.addLayout(action_row)

        # Run / Clear buttons
        self.run_label_btn = QPushButton("Run LLM Labeling on Current Document")
        self.run_label_btn.setObjectName("accentButton")
        self.run_label_btn.clicked.connect(self._run_labeling)
        layout.addWidget(self.run_label_btn)
        self.run_label_btn.hide()

        self.clear_labels_btn = QPushButton("Clear Labels for This Document")
        self.clear_labels_btn.setObjectName("dangerButton")
        self.clear_labels_btn.clicked.connect(self._clear_all_labels)
        layout.addWidget(self.clear_labels_btn)
        self.clear_labels_btn.hide()

    def _setup_shortcuts(self) -> None:
        # Left/Right arrows for prev/next chunk navigation, scoped to this widget.
        prev_sc = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        prev_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        prev_sc.activated.connect(self._go_prev)

        next_sc = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        next_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        next_sc.activated.connect(self._go_next)

    # --- Lifecycle -------------------------------------------------------

    def activate(self) -> None:
        self._reload()

    def on_document_selected(self, _doc_id: int) -> None:
        self._reload()

    def _current_document_id(self) -> int | None:
        return self.ctx.get_ui_state("selected_document_id")

    # --- Data loading ----------------------------------------------------

    def _reload(self) -> None:
        """Pull chunks/labels/reviews for the current doc and repaint."""
        if self.ctx.label_manager is None or self.ctx.rubric_manager is None:
            self._set_empty("Open a project first.")
            return

        rubric = self.ctx.rubric_manager.get_active_rubric()
        if rubric is None:
            self._set_empty("Save a rubric first.")
            return

        doc_id = self._current_document_id()
        if doc_id is None:
            self._set_empty("Select a document.")
            return

        if self.ctx.database is None:
            self._set_empty("No database connection.")
            return

        chunks = self.ctx.database.get_chunks_for_document(doc_id)
        self._chunks = chunks

        # Map chunk_id → latest label (for the active rubric only).
        self._labels_by_chunk = {}
        labels = self.ctx.database.get_all_labels(rubric.id)
        for lab in labels:
            self._labels_by_chunk[lab.chunk_id] = lab

        # Map label_id → latest review.
        self._reviews_by_label = {}
        for chunk in chunks:
            lab = self._labels_by_chunk.get(chunk.id or -1)
            if lab and lab.id is not None:
                reviews = self.ctx.database.get_reviews_for_label(lab.id)
                if reviews:
                    self._reviews_by_label[lab.id] = reviews[-1]

        self._configure_timeline_for_rubric(rubric)
        self._refresh_timeline_statuses()

        # Visibility of the corpus-level buttons.
        any_unlabeled = any(
            self._labels_by_chunk.get(c.id or -1) is None for c in chunks
        )
        any_labeled = any(
            self._labels_by_chunk.get(c.id or -1) is not None for c in chunks
        )
        self.run_label_btn.setVisible(any_unlabeled and bool(chunks))
        self.clear_labels_btn.setVisible(any_labeled)

        # Pin current_index in range.
        if not chunks:
            self._current_index = 0
            self._set_empty(
                "No chunks. Run chunking on this document first (Chunk tab)."
            )
            return

        self._current_index = max(0, min(self._current_index, len(chunks) - 1))
        self._show_current()

    def _configure_timeline_for_rubric(self, rubric: Rubric) -> None:
        """Set total frames + per-category color mapping on the timeline."""
        color_map: dict[str, QColor] = {}
        for i, cat in enumerate(rubric.categories):
            color_map[cat.name] = _category_color(i)
        self.timeline.timeline.set_custom_colors(color_map)
        self.timeline.timeline.set_frame_count(len(self._chunks))
        # Names for hover tooltip — chunk number + first 60 chars.
        names: list[str] = []
        for i, chunk in enumerate(self._chunks):
            preview = chunk.text.replace("\n", " ").strip()
            if len(preview) > 60:
                preview = preview[:60] + "…"
            names.append(f"#{i + 1}  {preview}")
        self.timeline.timeline.set_frame_names(names)

    def _refresh_timeline_statuses(self) -> None:
        statuses: dict[int, str] = {}
        confidences: dict[int, float] = {}
        for i, chunk in enumerate(self._chunks):
            lab = self._labels_by_chunk.get(chunk.id or -1)
            if not lab:
                continue
            review = (
                self._reviews_by_label.get(lab.id) if lab.id is not None else None
            )
            # Pick category to colour by: human-corrected > predicted.
            cats = (
                review.final_categories
                if review and review.final_categories
                else lab.predicted_categories
            )
            if cats:
                statuses[i] = cats[0]
            confidences[i] = lab.composite_confidence or 0.0
        self.timeline.timeline.set_batch_statuses(statuses)
        self.timeline.timeline.set_confidence_scores(confidences)
        if self._chunks:
            self.timeline.timeline.set_current_frame(self._current_index)

    # --- Render the current chunk ---------------------------------------

    def _set_empty(self, message: str) -> None:
        self._chunks = []
        self._labels_by_chunk = {}
        self._reviews_by_label = {}
        self.progress_label.setText(message)
        self.chunk_text.clear()
        self.source_label.clear()
        self.rationale_label.clear()
        self._clear_confidence_bars()
        self.timeline.timeline.set_frame_count(0)
        self.run_label_btn.setVisible(False)
        self.clear_labels_btn.setVisible(False)

    def _clear_confidence_bars(self) -> None:
        while self.confidence_layout.count():
            child = self.confidence_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _show_current(self) -> None:
        if not self._chunks:
            return
        idx = max(0, min(self._current_index, len(self._chunks) - 1))
        self._current_index = idx
        chunk = self._chunks[idx]
        lab = self._labels_by_chunk.get(chunk.id or -1)
        review = (
            self._reviews_by_label.get(lab.id)
            if lab and lab.id is not None
            else None
        )

        # Progress + label summary
        labeled_count = sum(
            1 for c in self._chunks if self._labels_by_chunk.get(c.id or -1)
        )
        if lab:
            cats = (
                review.final_categories
                if review and review.final_categories
                else lab.predicted_categories
            )
            cat_str = ", ".join(cats) if cats else "—"
            review_tag = f" · reviewed: {review.action}" if review else ""
            self.progress_label.setText(
                f"Chunk {idx + 1}/{len(self._chunks)}  ·  "
                f"{labeled_count}/{len(self._chunks)} labeled  ·  "
                f"{cat_str}  ·  {lab.composite_confidence:.0%}{review_tag}"
            )
        else:
            self.progress_label.setText(
                f"Chunk {idx + 1}/{len(self._chunks)}  ·  "
                f"{labeled_count}/{len(self._chunks)} labeled  ·  unlabeled"
            )

        # Confidence bars
        self._clear_confidence_bars()
        if lab:
            confidence_source = (
                {c: 1.0 for c in (review.final_categories or [])}
                if review and review.final_categories
                else lab.confidence_per_category
            )
            for cat, conf in sorted(confidence_source.items(), key=lambda x: -x[1]):
                bar = ConfidenceBar(value=conf, label=cat)
                self.confidence_layout.addWidget(bar)

        self.chunk_text.setPlainText(chunk.text)

        section = " > ".join(chunk.section_path) if chunk.section_path else ""
        self.source_label.setText(
            f"Source: chars {chunk.char_start}-{chunk.char_end}  ·  "
            f"{chunk.token_count} tokens  ·  {section}".strip(" ·")
        )

        if lab and lab.rationale:
            self.rationale_label.setText(f"LLM rationale: {lab.rationale}")
        else:
            self.rationale_label.clear()

        # Timeline current marker
        self.timeline.timeline.set_current_frame(idx)

        # Nav button enable state
        self.prev_btn.setEnabled(idx > 0)
        self.next_btn.setEnabled(idx < len(self._chunks) - 1)

    # --- Navigation -----------------------------------------------------

    def _on_timeline_select(self, frame_idx: int) -> None:
        if 0 <= frame_idx < len(self._chunks):
            self._current_index = frame_idx
            self._show_current()

    def _go_prev(self) -> None:
        if self._current_index > 0:
            self._current_index -= 1
            self._show_current()

    def _go_next(self) -> None:
        if self._current_index < len(self._chunks) - 1:
            self._current_index += 1
            self._show_current()

    # --- Review actions -------------------------------------------------

    def _current_label(self) -> Label | None:
        if not self._chunks:
            return None
        chunk = self._chunks[self._current_index]
        return self._labels_by_chunk.get(chunk.id or -1)

    def accept_current(self) -> None:
        self._submit_review("accept")

    def _skip_current(self) -> None:
        self._submit_review("skip")

    def _flag_current(self) -> None:
        self._submit_review("flag")

    def _correct_current(self) -> None:
        if self.ctx.rubric_manager is None or self.ctx.label_manager is None:
            return
        rubric = self.ctx.rubric_manager.get_active_rubric()
        if rubric is None:
            return
        lab = self._current_label()
        if lab is None or lab.id is None:
            self.main_window.notification_manager.show_warning(
                "Run labeling on this chunk first; nothing to correct yet."
            )
            return

        chosen = self._pick_categories(
            rubric.categories,
            preselected=list(lab.predicted_categories or []),
        )
        if chosen is None:
            return
        if not chosen:
            self.main_window.notification_manager.show_warning(
                "Pick at least one category, or use Discard."
            )
            return

        try:
            review = self.ctx.label_manager.submit_review(
                lab.id, "correct", final_categories=chosen
            )
            self._reviews_by_label[lab.id] = review
        except Exception as e:
            logger.error("Correction failed: %s", e)
            self.main_window.notification_manager.show_error(
                f"Correction failed: {e}"
            )
            return

        self._refresh_timeline_statuses()
        self._show_current()
        self.main_window._update_stats()
        self._advance_if_possible()

    def _submit_review(self, action: str) -> None:
        lab = self._current_label()
        if lab is None or lab.id is None:
            self.main_window.notification_manager.show_warning(
                "Nothing labeled here yet — run labeling first."
            )
            return
        if self.ctx.label_manager is None:
            return
        try:
            final_cats = (
                list(lab.predicted_categories) if action == "accept" else []
            )
            review = self.ctx.label_manager.submit_review(
                lab.id, action, final_categories=final_cats
            )
            self._reviews_by_label[lab.id] = review
        except Exception as e:
            logger.error("Review failed: %s", e)
            return

        self._refresh_timeline_statuses()
        self._show_current()
        self.main_window._update_stats()
        self._advance_if_possible()

    def _advance_if_possible(self) -> None:
        # Move to the next chunk that's labeled but not yet reviewed.
        for i in range(self._current_index + 1, len(self._chunks)):
            chunk = self._chunks[i]
            lab = self._labels_by_chunk.get(chunk.id or -1)
            if lab and (lab.id is None or lab.id not in self._reviews_by_label):
                self._current_index = i
                self._show_current()
                return
        # No further unreviewed labeled chunk; stay put.

    def _discard_current(self) -> None:
        lab = self._current_label()
        if lab is None or lab.id is None:
            return
        if self.ctx.label_manager is None:
            return
        try:
            self.ctx.label_manager.delete_label(lab.id)
        except Exception as e:
            self.main_window.notification_manager.show_error(
                f"Discard failed: {e}"
            )
            return

        chunk = self._chunks[self._current_index]
        if chunk.id is not None:
            self._labels_by_chunk.pop(chunk.id, None)
        self._reviews_by_label.pop(lab.id, None)

        self._refresh_timeline_statuses()
        any_unlabeled = any(
            self._labels_by_chunk.get(c.id or -1) is None for c in self._chunks
        )
        any_labeled = any(
            self._labels_by_chunk.get(c.id or -1) is not None for c in self._chunks
        )
        self.run_label_btn.setVisible(any_unlabeled and bool(self._chunks))
        self.clear_labels_btn.setVisible(any_labeled)
        self._show_current()
        self.main_window._update_stats()

    def _pick_categories(self, categories, preselected: list[str]) -> list[str] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Correct label")
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Select the correct categories:"))

        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for cat in categories:
            item = QListWidgetItem(cat.name)
            if cat.definition:
                item.setToolTip(cat.definition)
            list_widget.addItem(item)
            if cat.name in preselected:
                item.setSelected(True)
        layout.addWidget(list_widget)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return [item.text() for item in list_widget.selectedItems()]

    # --- Bulk operations ------------------------------------------------

    def _run_labeling(self) -> None:
        if self.ctx.label_manager is None or self.ctx.rubric_manager is None:
            return
        if self._labeling_worker is not None and self._labeling_worker.isRunning():
            return
        rubric = self.ctx.rubric_manager.get_active_rubric()
        if rubric is None:
            return
        doc_id = self._current_document_id()
        unlabeled = self.ctx.label_manager.get_unlabeled_chunks(
            rubric.id, document_id=doc_id
        )
        if not unlabeled:
            return

        self._labeling_total = len(unlabeled)
        self.run_label_btn.setEnabled(False)
        self.progress_label.setText(f"Labeling 0/{self._labeling_total} chunks...")

        worker = LabelingWorker(
            self.ctx.label_manager, unlabeled, rubric, parent=self
        )
        worker.progress.connect(self._on_labeling_progress)
        worker.chunk_labeled.connect(self._on_labeling_chunk_done)
        worker.finished.connect(self._on_labeling_finished)
        worker.error.connect(self._on_labeling_error)
        self._labeling_worker = worker
        worker.start()

    def _on_labeling_progress(self, current: int, total: int) -> None:
        self.progress_label.setText(f"Labeling chunk {current}/{total}...")

    def _on_labeling_chunk_done(self, _chunk_id: int) -> None:
        self._reload()

    def _on_labeling_finished(self) -> None:
        self.main_window.notification_manager.show_success(
            f"Labeled {self._labeling_total} chunks"
        )
        self.run_label_btn.setEnabled(True)
        self._labeling_worker = None
        self._reload()

    def _on_labeling_error(self, msg: str) -> None:
        self.main_window.notification_manager.show_error(f"Labeling failed: {msg}")
        self.run_label_btn.setEnabled(True)
        self._labeling_worker = None

    def _clear_all_labels(self) -> None:
        if self.ctx.label_manager is None:
            return
        doc_id = self._current_document_id()
        if doc_id is None:
            self.main_window.notification_manager.show_warning(
                "Select a document first."
            )
            return

        doc_name = "this document"
        if self.ctx.document_manager:
            try:
                doc = self.ctx.document_manager.get_document(doc_id)
                if doc:
                    doc_name = doc.filename
            except Exception:
                pass

        confirm = QMessageBox.question(
            self,
            "Clear labels for this document?",
            f"This will delete every label and review for '{doc_name}' "
            "across all rubric versions. The chunks themselves are kept; they "
            "will return to the unlabeled pool. This cannot be undone.\n\n"
            "To wipe the whole project, use Reset Project in the left panel.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            count = self.ctx.label_manager.clear_labels_for_document(doc_id)
            if self.ctx.audit_manager:
                self.ctx.audit_manager.log_event(
                    "labels_cleared_for_document",
                    payload={"document_id": doc_id, "count": count},
                )
            self.main_window.notification_manager.show_success(
                f"Cleared {count} labels for {doc_name}"
            )
        except Exception as e:
            self.main_window.notification_manager.show_error(
                f"Clear failed: {e}"
            )
            return
        self._reload()
        self.main_window._update_stats()

    # --- Misc -----------------------------------------------------------

    def handle_escape(self) -> None:
        pass
