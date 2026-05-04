"""Conversion worker: batch document conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from lazylabeltext.core.document_manager import DocumentManager


class ConversionWorker(QThread):
    """Background thread for batch document conversion."""

    progress = pyqtSignal(int, int)  # current, total
    document_converted = pyqtSignal(int)  # doc_id
    document_failed = pyqtSignal(int, str)  # doc_id, error
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        document_manager: DocumentManager,
        file_paths: list[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.document_manager = document_manager
        self.file_paths = file_paths
        self._should_stop = False

    def stop(self) -> None:
        self._should_stop = True

    def run(self) -> None:
        try:
            total = len(self.file_paths)
            for i, path in enumerate(self.file_paths):
                if self._should_stop:
                    return
                try:
                    doc = self.document_manager.load_single(path)
                    self.document_converted.emit(doc.id or 0)
                except Exception as e:
                    self.document_failed.emit(0, str(e))
                self.progress.emit(i + 1, total)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
