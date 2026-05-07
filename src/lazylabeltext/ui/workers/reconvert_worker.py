"""Reconvert worker: re-run conversion for a single existing document."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from lazylabeltext.core.document_manager import DocumentManager


class ReconvertWorker(QThread):
    """Background thread for re-running conversion on one document."""

    status_changed = pyqtSignal(str)
    finished_ok = pyqtSignal(int)  # new doc_id
    failed = pyqtSignal(str)

    def __init__(
        self,
        document_manager: DocumentManager,
        doc_id: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.document_manager = document_manager
        self.doc_id = doc_id

    def run(self) -> None:
        self.status_changed.emit("Reconverting…")
        try:
            doc = self.document_manager.reconvert(self.doc_id)
        except Exception as e:
            self.failed.emit(str(e))
            return
        if doc is None:
            self.failed.emit("Source file not found.")
            return
        self.finished_ok.emit(doc.id or 0)
