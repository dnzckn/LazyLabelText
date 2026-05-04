"""Labeling worker: batch LLM labeling in background."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from lazylabeltext.core.label_manager import LabelManager
    from lazylabeltext.core.models import Chunk, Rubric


class LabelingWorker(QThread):
    """Background thread for LLM labeling."""

    progress = pyqtSignal(int, int)  # current, total
    chunk_labeled = pyqtSignal(int)  # chunk_id
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        label_manager: LabelManager,
        chunks: list[Chunk],
        rubric: Rubric,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.label_manager = label_manager
        self.chunks = list(chunks)  # Snapshot
        self.rubric = rubric
        self._should_stop = False

    def stop(self) -> None:
        self._should_stop = True

    def run(self) -> None:
        try:
            for i, chunk in enumerate(self.chunks):
                if self._should_stop:
                    return
                self.label_manager.label_chunk(chunk, self.rubric)
                self.chunk_labeled.emit(chunk.id or 0)
                self.progress.emit(i + 1, len(self.chunks))
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
