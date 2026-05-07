"""Conversion worker: batch document conversion (optionally parallel)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from lazylabeltext.core.document_manager import DocumentManager


class ConversionWorker(QThread):
    """Background thread for batch document conversion.

    With ``max_workers > 1`` documents are converted in a thread pool. The
    default is 1 because docling holds a singleton model and parallel
    conversion can spike memory; lightweight converters (PyMuPDF, docx,
    markdown) safely handle 4+ concurrent jobs.
    """

    progress = pyqtSignal(int, int)  # current, total
    status_changed = pyqtSignal(str)  # human-readable status
    document_converted = pyqtSignal(int)  # doc_id
    document_failed = pyqtSignal(int, str)  # doc_id, error
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        document_manager: DocumentManager,
        file_paths: list[str],
        max_workers: int = 1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.document_manager = document_manager
        self.file_paths = file_paths
        self.max_workers = max(1, int(max_workers))
        self._should_stop = False

    def stop(self) -> None:
        self._should_stop = True

    def _convert_one(self, path: str) -> tuple[str, int | None, str | None]:
        """Run a single conversion. Returns (path, doc_id_or_None, error_or_None)."""
        try:
            doc = self.document_manager.load_single(path)
            return path, doc.id or 0, None
        except Exception as e:
            failed = self.document_manager.record_failed(path, str(e))
            return path, failed.id or 0, str(e)

    def run(self) -> None:
        try:
            total = len(self.file_paths)
            if total == 0:
                self.finished.emit()
                return

            if self.max_workers == 1:
                # Sequential path keeps the trivial-case stack trace clean
                # and avoids the executor for projects with one or two docs.
                for i, path in enumerate(self.file_paths):
                    if self._should_stop:
                        break
                    self.status_changed.emit(f"Converting {Path(path).name}…")
                    _, doc_id, err = self._convert_one(path)
                    if err is None:
                        self.document_converted.emit(doc_id or 0)
                    else:
                        self.document_failed.emit(doc_id or 0, err)
                    self.progress.emit(i + 1, total)
            else:
                # Parallel path. Per-thread DB connections + the class-wide
                # RLock on Database make concurrent inserts safe; converters
                # themselves are pure (PyMuPDF/docx) or singleton-guarded
                # (docling) on the read side.
                in_flight_lock = threading.Lock()
                in_flight = {"current": ""}

                def _emit_status_for(path: str) -> None:
                    with in_flight_lock:
                        in_flight["current"] = Path(path).name
                        self.status_changed.emit(
                            f"Converting {in_flight['current']}…"
                        )

                completed = 0
                with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                    futs = {}
                    for path in self.file_paths:
                        _emit_status_for(path)
                        futs[ex.submit(self._convert_one, path)] = path
                    for fut in as_completed(futs):
                        if self._should_stop:
                            ex.shutdown(wait=False, cancel_futures=True)
                            break
                        try:
                            _, doc_id, err = fut.result()
                        except Exception as e:
                            # Defensive — _convert_one already catches.
                            self.document_failed.emit(0, str(e))
                            err = str(e)
                            doc_id = 0
                        if err is None:
                            self.document_converted.emit(doc_id or 0)
                        else:
                            self.document_failed.emit(doc_id or 0, err)
                        completed += 1
                        self.progress.emit(completed, total)

            self.status_changed.emit("")
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
