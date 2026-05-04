"""Export worker: background export operations."""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from lazylabeltext.core.exporters import ExportFormat, export_corpus


class ExportWorker(QThread):
    """Background thread for corpus export."""

    finished = pyqtSignal(str)  # output_path
    error = pyqtSignal(str)

    def __init__(
        self,
        db,
        fmt: ExportFormat,
        rubric_version_id: int,
        output_path: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.db = db
        self.fmt = fmt
        self.rubric_version_id = rubric_version_id
        self.output_path = output_path

    def run(self) -> None:
        try:
            path = export_corpus(
                self.fmt, self.db, self.rubric_version_id, self.output_path
            )
            self.finished.emit(path)
        except Exception as e:
            self.error.emit(str(e))
