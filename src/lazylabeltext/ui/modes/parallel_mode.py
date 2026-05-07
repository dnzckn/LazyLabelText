"""Parallel Mode: corpus-wide stage runner with per-doc drill-in.

Layout: a compact left strip (~280px) with the doc list and stage controls;
the rest of the screen hosts an inner tab bar (Convert / Chunk / Label)
that embeds fresh instances of the existing single-doc viewers, so drilling
into a doc reuses the same UIs the user already knows.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.core.chunkers import available_strategies
from lazylabeltext.core.parallel_orchestrator import DocState
from lazylabeltext.ui.modes.base_mode import BaseMode
from lazylabeltext.ui.modes.chunk_mode import ChunkModeWidget
from lazylabeltext.ui.modes.convert_mode import ConvertModeWidget
from lazylabeltext.ui.modes.label_mode import LabelModeWidget
from lazylabeltext.ui.widgets.parallel_doc_strip import ParallelDocStrip
from lazylabeltext.ui.workers.parallel_worker import ParallelWorker

if TYPE_CHECKING:
    from lazylabeltext.core.app_context import AppContext
    from lazylabeltext.ui.main_window import MainWindow

logger = logging.getLogger("lazylabeltext")


_STAGE_OPTIONS = [
    ("Convert", "convert"),
    ("Chunk", "chunk"),
    ("Label", "label"),
    ("All Stages", "all"),
]


class ParallelModeWidget(BaseMode):
    def __init__(
        self,
        context: AppContext,
        main_window: MainWindow,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(context, parent)
        self.main_window = main_window
        self._worker: ParallelWorker | None = None
        self._setup_ui()

    # --- UI setup --------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: control strip — splitter-resizable (no hard maximum).
        # Default ~320px so the stage settings render without clipping; the
        # user can drag the splitter handle wider if they want more room.
        left = QWidget()
        left.setMinimumWidth(220)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 6, 6, 6)
        left_layout.setSpacing(6)

        header = QLabel("Parallel")
        header.setObjectName("sectionHeader")
        left_layout.addWidget(header)

        # Stage selector
        stage_row = QHBoxLayout()
        stage_row.setSpacing(4)
        stage_row.addWidget(QLabel("Stage:"))
        self.stage_combo = QComboBox()
        for label, _ in _STAGE_OPTIONS:
            self.stage_combo.addItem(label)
        self.stage_combo.currentIndexChanged.connect(self._on_stage_changed)
        stage_row.addWidget(self.stage_combo, 1)
        left_layout.addLayout(stage_row)

        # Workers
        workers_row = QHBoxLayout()
        workers_row.setSpacing(4)
        workers_row.addWidget(QLabel("Workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setToolTip(
            "Concurrent worker threads. Effective parallelism is also bounded by "
            "your LLM provider — e.g. Anthropic free tier serializes to 1 request "
            "at a time and Ollama serializes per loaded model regardless of this value."
        )
        default_workers = (
            getattr(self.ctx.settings, "parallel_workers", 4)
            if self.ctx.settings is not None
            else 4
        )
        self.workers_spin.setValue(default_workers)
        self.workers_spin.valueChanged.connect(self._on_workers_changed)
        workers_row.addWidget(self.workers_spin, 1)
        left_layout.addLayout(workers_row)

        # Stage settings — three group boxes, visibility driven by stage selector.
        # Edits here mutate Settings directly and persist on every change so the
        # orchestrator (which reads from settings via _build_params) sees them.
        self.convert_group = self._build_convert_settings()
        self.chunk_group = self._build_chunk_settings()
        self.label_group = self._build_label_settings()
        left_layout.addWidget(self.convert_group)
        left_layout.addWidget(self.chunk_group)
        left_layout.addWidget(self.label_group)
        self._on_stage_changed(self.stage_combo.currentIndex())

        # Run / Cancel buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.run_btn = QPushButton("▶ Run")
        self.run_btn.setObjectName("accentButton")
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)
        self.cancel_btn = QPushButton("■ Cancel")
        self.cancel_btn.setObjectName("dangerButton")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.cancel_btn)
        left_layout.addLayout(btn_row)

        # Doc strip
        self.strip = ParallelDocStrip()
        self.strip.selection_changed.connect(self._on_strip_selection)
        self.strip.inclusion_toggled.connect(self._on_inclusion_toggled)
        self.strip.rerun_requested.connect(self._on_rerun_requested)
        self.strip.drop_requested.connect(self._on_drop_requested)
        left_layout.addWidget(self.strip, 1)

        # Stage status line
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        splitter.addWidget(left)

        # Right: drill-in tabs hosting the existing single-doc widgets
        self.detail_tabs = QTabWidget()
        self.convert_widget = ConvertModeWidget(self.ctx)
        self.chunk_widget = ChunkModeWidget(self.ctx, self.main_window)
        self.label_widget = LabelModeWidget(self.ctx, self.main_window)
        self.detail_tabs.addTab(self.convert_widget, "Convert")
        self.detail_tabs.addTab(self.chunk_widget, "Chunk")
        self.detail_tabs.addTab(self.label_widget, "Label")
        splitter.addWidget(self.detail_tabs)

        # Allow the user to grow the strip by dragging the splitter handle.
        # Both sides stretch (factor=1) so manually-set sizes are respected
        # and the strip expands proportionally on window resize. The default
        # split favors the viewers but the strip can grow up to half.
        splitter.setSizes([320, 1080])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter)

    # --- Stage settings panels -------------------------------------------

    def _settings(self):
        return self.ctx.settings

    def _persist_settings(self) -> None:
        if self.ctx.settings is None or self.ctx.paths is None:
            return
        with contextlib.suppress(Exception):
            self.ctx.settings.save_to_file(str(self.ctx.paths.settings_file))

    def _build_convert_settings(self) -> QGroupBox:
        box = QGroupBox("Convert settings")
        box.setStyleSheet("QGroupBox { font-size: 11px; }")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)

        self.high_fidelity_check = QCheckBox("High-fidelity (docling)")
        self.high_fidelity_check.setToolTip(
            "Better tables/figures via docling. ~1–2 GB models on first run."
        )
        s = self._settings()
        self.high_fidelity_check.setChecked(
            getattr(s, "use_high_fidelity_conversion", False) if s else False
        )
        self.high_fidelity_check.toggled.connect(self._on_high_fidelity_toggled)
        v.addWidget(self.high_fidelity_check)

        self.ocr_check = QCheckBox("OCR (scanned PDFs)")
        self.ocr_check.setToolTip("Requires high-fidelity mode.")
        self.ocr_check.setChecked(
            getattr(s, "high_fidelity_ocr", False) if s else False
        )
        self.ocr_check.setEnabled(self.high_fidelity_check.isChecked())
        self.ocr_check.toggled.connect(self._on_ocr_toggled)
        v.addWidget(self.ocr_check)
        return box

    def _build_chunk_settings(self) -> QGroupBox:
        box = QGroupBox("Chunk settings")
        box.setStyleSheet("QGroupBox { font-size: 11px; }")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)

        s = self._settings()

        strat_row = QHBoxLayout()
        strat_row.setSpacing(4)
        strat_row.addWidget(QLabel("Strategy:"))
        self.strategy_combo = QComboBox()
        strategies = available_strategies()
        self.strategy_combo.addItems(strategies)
        current = getattr(s, "default_chunk_strategy", "structural") if s else "structural"
        if current in strategies:
            self.strategy_combo.setCurrentText(current)
        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        strat_row.addWidget(self.strategy_combo, 1)
        v.addLayout(strat_row)

        size_row = QHBoxLayout()
        size_row.setSpacing(4)
        size_row.addWidget(QLabel("Min:"))
        self.min_tokens_spin = QSpinBox()
        self.min_tokens_spin.setRange(10, 5000)
        self.min_tokens_spin.setValue(getattr(s, "chunk_min_tokens", 50) if s else 50)
        self.min_tokens_spin.valueChanged.connect(self._on_min_tokens_changed)
        size_row.addWidget(self.min_tokens_spin, 1)

        size_row.addWidget(QLabel("Max:"))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(50, 20000)
        self.max_tokens_spin.setValue(getattr(s, "chunk_max_tokens", 800) if s else 800)
        self.max_tokens_spin.valueChanged.connect(self._on_max_tokens_changed)
        size_row.addWidget(self.max_tokens_spin, 1)
        v.addLayout(size_row)

        levels_row = QHBoxLayout()
        levels_row.setSpacing(2)
        levels_row.addWidget(QLabel("Heading levels:"))
        active_levels = set(
            getattr(s, "heading_split_levels", [1, 2, 3]) if s else [1, 2, 3]
        )
        self.level_checks: dict[int, QCheckBox] = {}
        for lvl in (1, 2, 3, 4, 5, 6):
            cb = QCheckBox(str(lvl))
            cb.setChecked(lvl in active_levels)
            cb.toggled.connect(self._on_levels_changed)
            levels_row.addWidget(cb)
            self.level_checks[lvl] = cb
        levels_row.addStretch(1)
        v.addLayout(levels_row)
        return box

    def _build_label_settings(self) -> QGroupBox:
        box = QGroupBox("Label settings")
        box.setStyleSheet("QGroupBox { font-size: 11px; }")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)

        self.rubric_status_label = QLabel()
        self.rubric_status_label.setWordWrap(True)
        self.rubric_status_label.setStyleSheet("color: #888; font-size: 11px;")
        v.addWidget(self.rubric_status_label)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #333;")
        v.addWidget(line)

        hint = QLabel(
            "Edit rubric in Rubric mode. Parallel labeling uses the active version."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 10px;")
        v.addWidget(hint)
        self._refresh_rubric_status()
        return box

    def _refresh_rubric_status(self) -> None:
        if self.ctx.rubric_manager is None:
            self.rubric_status_label.setText("(no rubric manager)")
            return
        rubric = self.ctx.rubric_manager.get_active_rubric()
        if rubric is None:
            self.rubric_status_label.setText("⚠ No active rubric — create one first.")
        else:
            self.rubric_status_label.setText(
                f"Active rubric: {rubric.name} v{rubric.version} "
                f"({len(rubric.categories)} categories)"
            )

    def _on_stage_changed(self, idx: int) -> None:
        stage = _STAGE_OPTIONS[idx][1]
        show_all = stage == "all"
        self.convert_group.setVisible(show_all or stage == "convert")
        self.chunk_group.setVisible(show_all or stage == "chunk")
        self.label_group.setVisible(show_all or stage == "label")
        if show_all or stage == "label":
            self._refresh_rubric_status()
        self._sync_detail_tab_to_stage()

    def _sync_detail_tab_to_stage(self) -> None:
        """Switch the right-side viewer tab to match the selected stage.

        Convert/Chunk/Label map 1:1 to the inner tabs. "All Stages" leaves
        whichever tab the user last had — there's no single "all" view to
        switch to. The user can still manually change tabs after this fires.
        """
        # Initial _on_stage_changed call from _setup_ui fires before the
        # detail_tabs widget is built — guard against that.
        if not hasattr(self, "detail_tabs"):
            return
        stage = _STAGE_OPTIONS[self.stage_combo.currentIndex()][1]
        tab_idx = {"convert": 0, "chunk": 1, "label": 2}.get(stage)
        if tab_idx is not None:
            self.detail_tabs.setCurrentIndex(tab_idx)

    def _on_workers_changed(self, n: int) -> None:
        s = self._settings()
        if s is not None:
            s.parallel_workers = n
        self._persist_settings()

    def _on_high_fidelity_toggled(self, checked: bool) -> None:
        s = self._settings()
        if s is not None:
            s.use_high_fidelity_conversion = checked
            if not checked:
                s.high_fidelity_ocr = False
                self.ocr_check.blockSignals(True)
                self.ocr_check.setChecked(False)
                self.ocr_check.blockSignals(False)
        self.ocr_check.setEnabled(checked)
        self._persist_settings()

    def _on_ocr_toggled(self, checked: bool) -> None:
        s = self._settings()
        if s is not None:
            s.high_fidelity_ocr = checked
        self._persist_settings()

    def _on_strategy_changed(self, text: str) -> None:
        s = self._settings()
        if s is not None:
            s.default_chunk_strategy = text
        self._persist_settings()

    def _on_min_tokens_changed(self, v: int) -> None:
        s = self._settings()
        if s is not None:
            s.chunk_min_tokens = v
        self._persist_settings()

    def _on_max_tokens_changed(self, v: int) -> None:
        s = self._settings()
        if s is not None:
            s.chunk_max_tokens = v
        self._persist_settings()

    def _on_levels_changed(self, _checked: bool) -> None:
        s = self._settings()
        if s is None:
            return
        s.heading_split_levels = [
            lvl for lvl, cb in self.level_checks.items() if cb.isChecked()
        ]
        self._persist_settings()

    # --- BaseMode hooks --------------------------------------------------

    def activate(self) -> None:
        if self.ctx.parallel_orchestrator is None:
            return
        states = self.ctx.parallel_orchestrator.refresh_states()
        self.strip.populate(states)
        self._refresh_rubric_status()

    # --- Run / cancel ----------------------------------------------------

    def _selected_stage(self) -> str:
        idx = self.stage_combo.currentIndex()
        return _STAGE_OPTIONS[idx][1]

    def _build_params(self) -> dict:
        s = self.ctx.settings
        params: dict = {}
        # Chunking pulls strategy + per-strategy params from settings (the
        # same ones the user tuned in chunk mode). Never hard-coded here.
        if s is not None:
            params["strategy"] = getattr(s, "default_chunk_strategy", "structural")
            params["chunk_params"] = {
                "min_tokens": getattr(s, "chunk_min_tokens", 50),
                "max_tokens": getattr(s, "chunk_max_tokens", 800),
                "heading_split_levels": getattr(
                    s, "heading_split_levels", [1, 2, 3]
                ),
            }
        # Labeling needs the active rubric.
        if self.ctx.rubric_manager is not None:
            params["rubric"] = self.ctx.rubric_manager.get_active_rubric()
        return params

    def _included_doc_ids(self) -> list[int]:
        if self.ctx.parallel_orchestrator is None:
            return []
        return [s.doc_id for s in self.ctx.parallel_orchestrator.get_states() if s.included]

    def _on_run(self) -> None:
        if self.ctx.parallel_orchestrator is None:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        stage = self._selected_stage()
        doc_ids = self._included_doc_ids()
        if not doc_ids:
            self.status_label.setText("No documents selected.")
            return
        params = self._build_params()
        if stage in ("label", "all") and not params.get("rubric"):
            self.status_label.setText("No active rubric — create one first.")
            return

        op = "all_stages" if stage == "all" else "stage"
        worker = ParallelWorker(
            self.ctx.parallel_orchestrator,
            op=op,
            stage=None if stage == "all" else stage,  # type: ignore[arg-type]
            doc_ids=doc_ids,
            params=params,
            max_workers=self.workers_spin.value(),
            parent=self,
        )
        self._wire_worker(worker)
        self._worker = worker
        self._set_running(True)
        worker.start()

    def _on_cancel(self) -> None:
        if self._worker is None:
            return
        self._worker.request_cancel()
        self.status_label.setText("Cancel requested — waiting for in-flight work to finish…")

    # --- Strip event handlers --------------------------------------------

    def _on_strip_selection(self, doc_id: int) -> None:
        if doc_id <= 0:
            return
        # Update global ui-state FIRST: label_mode's on_document_selected
        # ignores its doc_id argument and reads from ctx.get_ui_state(
        # "selected_document_id") instead. If we updated ui-state after
        # firing on_document_selected, label_mode would re-render the
        # *previous* doc, not the one the user just clicked.
        self.ctx.set_ui_state("selected_document_id", doc_id)
        # Push selection into each embedded viewer so the drill-in shows
        # the right doc regardless of which inner tab is active.
        for w in (self.convert_widget, self.chunk_widget, self.label_widget):
            try:
                w.on_document_selected(doc_id)
            except Exception:
                logger.debug("Embedded viewer rejected doc_id %s", doc_id, exc_info=True)
        # Default the detail tab to whichever stage is currently selected so
        # clicking a doc lands on the most relevant viewer for the current
        # task. The user can still manually switch tabs afterwards.
        self._sync_detail_tab_to_stage()

    def _on_inclusion_toggled(self, doc_id: int, included: bool) -> None:
        if self.ctx.parallel_orchestrator is None:
            return
        s = self.ctx.parallel_orchestrator.set_inclusion(doc_id, included)
        if s is not None:
            self.strip.update_doc(s)

    def _on_drop_requested(self, doc_id: int) -> None:
        if self.ctx.parallel_orchestrator is None:
            return
        # Same as un-checking, but also stops in-flight work as soon as
        # the orchestrator hits its next safe checkpoint.
        s = self.ctx.parallel_orchestrator.set_inclusion(doc_id, False)
        if s is not None:
            self.strip.update_doc(s)

    def _on_rerun_requested(self, doc_id: int, stage: str) -> None:
        if self.ctx.parallel_orchestrator is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self.status_label.setText(
                "A run is in progress — wait for it to finish before re-running a single doc."
            )
            return
        params = self._build_params()
        if stage == "label" and not params.get("rubric"):
            self.status_label.setText("No active rubric — create one first.")
            return
        worker = ParallelWorker(
            self.ctx.parallel_orchestrator,
            op="rerun_one",
            stage=stage,  # type: ignore[arg-type]
            params=params,
            max_workers=1,
            single_doc_id=doc_id,
            parent=self,
        )
        self._wire_worker(worker)
        self._worker = worker
        self._set_running(True)
        worker.start()

    # --- Worker plumbing -------------------------------------------------

    def _wire_worker(self, worker: ParallelWorker) -> None:
        worker.doc_state_changed.connect(self._on_doc_state)
        worker.stage_started.connect(self._on_stage_started)
        worker.stage_progress.connect(self._on_stage_progress)
        worker.stage_finished.connect(self._on_stage_finished)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(self._on_worker_finished)

    def _on_doc_state(self, _doc_id: int, state_dict: dict) -> None:
        state = DocState(**state_dict)
        self.strip.update_doc(state)
        # Live-update the side panel: if this doc is the one currently
        # selected in the strip, refresh the embedded label viewer so its
        # timeline ticks as new chunks complete instead of forcing the
        # user to switch docs back and forth to pick up changes. Convert
        # and chunk views don't change during labeling so we skip them.
        selected = self.strip.selected_doc_id()
        if selected and selected == state.doc_id and state.stage in (
            "label", "queued", "done", "cancelled", "failed",
        ):
            try:
                self.label_widget.on_document_selected(state.doc_id)
            except Exception:
                logger.debug(
                    "Live label refresh failed for doc %s", state.doc_id,
                    exc_info=True,
                )

    def _on_stage_started(self, stage: str) -> None:
        self.status_label.setText(f"Running stage: {stage}")

    def _on_stage_progress(self, current: int, total: int, message: str) -> None:
        self.status_label.setText(f"{message} ({current}/{total})")

    def _on_stage_finished(self, stage: str, summary: dict) -> None:
        succeeded = summary.get("succeeded", 0)
        failed = summary.get("failed", 0)
        artifacts = summary.get("artifacts", 0)
        cancelled = summary.get("cancelled", False)
        msg = f"{stage}: {succeeded} ok, {failed} failed, {artifacts} artifacts"
        if cancelled:
            msg += " (cancelled)"
        self.status_label.setText(msg)
        self.main_window._update_stats()
        # Belt-and-suspenders: rebuild the strip from the orchestrator's
        # final state. If any per-doc state-changed signal got dropped or
        # arrived out of order, this snapshot makes the strip consistent
        # with the orchestrator (which is the source of truth).
        if self.ctx.parallel_orchestrator is not None:
            self.strip.populate(self.ctx.parallel_orchestrator.get_states())

    def _on_worker_error(self, msg: str) -> None:
        self.status_label.setText(f"Error: {msg}")
        self.main_window.notification_manager.show_error(f"Parallel run failed: {msg}")

    def _on_worker_finished(self) -> None:
        self._set_running(False)
        self._worker = None

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.stage_combo.setEnabled(not running)
        self.workers_spin.setEnabled(not running)
