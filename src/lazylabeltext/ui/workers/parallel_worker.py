"""Parallel-mode worker: QThread shell around ParallelOrchestrator runs."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

from lazylabeltext.core.parallel_orchestrator import Callbacks

if TYPE_CHECKING:
    from lazylabeltext.core.parallel_orchestrator import (
        ParallelOrchestrator,
        Stage,
    )


class ParallelWorker(QThread):
    """Runs a single stage (or all stages, or a single-doc rerun) off the GUI thread."""

    doc_state_changed = pyqtSignal(int, dict)        # doc_id, state-as-dict
    stage_started = pyqtSignal(str)                   # stage name
    stage_progress = pyqtSignal(int, int, str)        # current, total, message
    stage_finished = pyqtSignal(str, dict)            # stage name, summary
    error = pyqtSignal(str)

    def __init__(
        self,
        orchestrator: ParallelOrchestrator,
        op: str,                # "stage" | "all_stages" | "rerun_one"
        stage: Stage | None = None,
        doc_ids: list[int] | None = None,
        params: dict | None = None,
        max_workers: int = 4,
        single_doc_id: int | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.op = op
        self.stage = stage
        self.doc_ids = doc_ids or []
        self.params = params or {}
        self.max_workers = max_workers
        self.single_doc_id = single_doc_id

    # Cancellation routes through the orchestrator's flag.
    def request_cancel(self) -> None:
        self.orchestrator.cancel_all()

    def _callbacks(self) -> Callbacks:
        return Callbacks(
            on_doc_state=lambda s: self.doc_state_changed.emit(s.doc_id, asdict(s)),
            on_stage_progress=lambda c, t, m: self.stage_progress.emit(c, t, m),
        )

    def run(self) -> None:
        try:
            if self.op == "stage":
                if self.stage is None:
                    self.error.emit("ParallelWorker: stage missing for op=stage")
                    return
                self.stage_started.emit(self.stage)
                result = self.orchestrator.run_stage(
                    self.stage, self.doc_ids, self.params,
                    self.max_workers, self._callbacks(),
                )
                self.stage_finished.emit(self.stage, asdict(result))
            elif self.op == "all_stages":
                cbs = self._callbacks()
                results = self.orchestrator.run_all_stages(
                    self.doc_ids, self.params, self.max_workers, cbs,
                )
                # Emit a synthetic 'all' summary plus per-stage summaries
                for r in results:
                    self.stage_started.emit(r.stage)
                    self.stage_finished.emit(r.stage, asdict(r))
            elif self.op == "rerun_one":
                if self.single_doc_id is None or self.stage is None:
                    self.error.emit("ParallelWorker: rerun_one missing doc_id/stage")
                    return
                self.stage_started.emit(self.stage)
                result = self.orchestrator.rerun_one(
                    self.single_doc_id, self.stage, self.params, self._callbacks(),
                )
                self.stage_finished.emit(self.stage, asdict(result))
            else:
                self.error.emit(f"ParallelWorker: unknown op {self.op!r}")
        except Exception as e:
            self.error.emit(str(e))
