"""Propagation worker: batch chunk + label across documents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from lazylabeltext.core.models import Rubric
    from lazylabeltext.core.propagation_manager import PropagationManager


class PropagationWorker(QThread):
    """Background thread for batch propagation."""

    progress = pyqtSignal(int, int, str)  # current, total, message
    finished_result = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        propagation_manager: PropagationManager,
        strategy: str,
        params: dict,
        rubric: Rubric,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.propagation_manager = propagation_manager
        self.strategy = strategy
        self.params = params
        self.rubric = rubric
        self._should_stop = False

    def cancel(self) -> None:
        self._should_stop = True

    def run(self) -> None:
        try:
            results = self.propagation_manager.propagate_all(
                chunking_strategy=self.strategy,
                chunking_params=self.params,
                rubric=self.rubric,
                progress_callback=lambda cur, total, msg: self.progress.emit(
                    cur, total, msg
                ),
                should_stop=lambda: self._should_stop,
            )
            self.finished_result.emit(results)
        except Exception as e:
            self.error.emit(str(e))
