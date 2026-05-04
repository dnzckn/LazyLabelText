"""Chunking worker: background chunking operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from lazylabeltext.core.chunk_manager import ChunkManager


class ChunkingWorker(QThread):
    """Background thread for chunking a document."""

    finished = pyqtSignal(int)  # run_id
    error = pyqtSignal(str)

    def __init__(
        self,
        chunk_manager: ChunkManager,
        document_id: int,
        strategy: str,
        params: dict,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.chunk_manager = chunk_manager
        self.document_id = document_id
        self.strategy = strategy
        self.params = params

    def run(self) -> None:
        try:
            run = self.chunk_manager.run_chunking(
                self.document_id, self.strategy, self.params
            )
            self.finished.emit(run.id or 0)
        except Exception as e:
            self.error.emit(str(e))
