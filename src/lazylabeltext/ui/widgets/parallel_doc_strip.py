"""Parallel-mode left strip: per-doc status, inclusion toggle, drill-in.

Compact widget designed to fit ~280px wide. Each row shows the doc's
filename, an include checkbox, a stage badge, an inline progress bar
(visible only mid-stage), and supports right-click for per-doc actions
(re-run convert/chunk/label here, drop from run).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from lazylabeltext.core.parallel_orchestrator import DocState


_STAGE_COLORS = {
    "idle": "#888",
    "queued": "#7a7a8f",
    "convert": "#5c8fbf",
    "chunk": "#7faf5c",
    "label": "#bf8f5c",
    "done": "#5cbf5c",
    "failed": "#bf5c5c",
    "cancelled": "#bfbf5c",
    "excluded": "#666",
}

_STAGE_LABELS = {
    "idle": "—",
    "queued": "queued",
    "convert": "converting",
    "chunk": "chunking",
    "label": "labeling",
    "done": "done",
    "failed": "failed",
    "cancelled": "cancelled",
    "excluded": "excluded",
}


class _DocRow(QWidget):
    """Single row in the strip — owns its checkbox / badge / progress widgets."""

    def __init__(self, state: DocState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc_id = state.doc_id

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(6)

        self.check = QCheckBox()
        self.check.setChecked(state.included)
        top.addWidget(self.check)

        self.name_label = QLabel(state.filename)
        self.name_label.setStyleSheet("font-size: 12px;")
        self.name_label.setToolTip(state.filename)
        top.addWidget(self.name_label, 1)

        self.badge = QLabel()
        self.badge.setStyleSheet("font-size: 10px;")
        top.addWidget(self.badge)

        outer.addLayout(top)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        outer.addWidget(self.progress)

        self.message_label = QLabel()
        self.message_label.setStyleSheet("color: #888; font-size: 10px;")
        outer.addWidget(self.message_label)

        self.update_state(state)

    def update_state(self, state: DocState) -> None:
        self.check.setChecked(state.included)
        # Unchecked docs always show "excluded" — the prior stage is
        # irrelevant once the user has opted them out of the next run.
        stage = "excluded" if not state.included else state.stage
        color = _STAGE_COLORS.get(stage, "#888")
        label = _STAGE_LABELS.get(stage, stage)
        self.badge.setText(f"●{label}")
        self.badge.setStyleSheet(f"color: {color}; font-size: 10px;")

        # Show progress only while actually working a doc — queued/idle
        # don't have meaningful progress yet.
        is_running = state.stage in ("convert", "chunk", "label")
        self.progress.setVisible(is_running)
        if is_running:
            self.progress.setValue(int(round(state.progress * 100)))

        msg = state.message
        if state.error and state.stage == "failed":
            msg = state.error[:80]
        if state.chunks or state.labels:
            counts = []
            if state.chunks:
                counts.append(f"{state.chunks}c")
            if state.labels:
                counts.append(f"{state.labels}l")
            joiner = " · "
            tail = joiner.join(counts)
            msg = f"{msg}  {tail}" if msg else tail
        self.message_label.setText(msg)


class ParallelDocStrip(QListWidget):
    """List of doc rows with inclusion + drill-in + per-doc actions."""

    selection_changed = pyqtSignal(int)             # doc_id (or 0 if cleared)
    inclusion_toggled = pyqtSignal(int, bool)
    rerun_requested = pyqtSignal(int, str)          # doc_id, stage
    drop_requested = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: dict[int, _DocRow] = {}
        self._items: dict[int, QListWidgetItem] = {}
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        self.itemSelectionChanged.connect(self._on_selection_changed)

    def populate(self, states: list[DocState]) -> None:
        self.clear()
        self._rows.clear()
        self._items.clear()
        for state in states:
            self._add_row(state)

    def _add_row(self, state: DocState) -> None:
        item = QListWidgetItem()
        row = _DocRow(state)
        row.check.toggled.connect(
            lambda checked, doc_id=state.doc_id: self.inclusion_toggled.emit(
                doc_id, checked
            )
        )
        item.setSizeHint(row.sizeHint())
        item.setData(Qt.ItemDataRole.UserRole, state.doc_id)
        self.addItem(item)
        self.setItemWidget(item, row)
        self._rows[state.doc_id] = row
        self._items[state.doc_id] = item

    def update_doc(self, state: DocState) -> None:
        row = self._rows.get(state.doc_id)
        if row is None:
            return
        row.update_state(state)
        item = self._items.get(state.doc_id)
        if item is not None:
            item.setSizeHint(row.sizeHint())

    def selected_doc_id(self) -> int | None:
        items = self.selectedItems()
        if not items:
            return None
        return items[0].data(Qt.ItemDataRole.UserRole)

    def select_doc(self, doc_id: int) -> None:
        item = self._items.get(doc_id)
        if item is not None:
            self.setCurrentItem(item)

    # ---- internal --------------------------------------------------------

    def _on_selection_changed(self) -> None:
        doc_id = self.selected_doc_id()
        self.selection_changed.emit(int(doc_id) if doc_id else 0)

    def _on_context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is None:
            return
        doc_id = int(item.data(Qt.ItemDataRole.UserRole))

        menu = QMenu(self)
        for stage_label, stage_key in (
            ("Re-run Convert here", "convert"),
            ("Re-run Chunk here", "chunk"),
            ("Re-run Label here", "label"),
        ):
            act = QAction(stage_label, menu)
            act.triggered.connect(
                lambda _checked=False, s=stage_key, d=doc_id: self.rerun_requested.emit(d, s)
            )
            menu.addAction(act)
        menu.addSeparator()
        drop = QAction("Drop from run (cancel this doc)", menu)
        drop.triggered.connect(lambda _checked=False, d=doc_id: self.drop_requested.emit(d))
        menu.addAction(drop)
        menu.exec(self.mapToGlobal(pos))
