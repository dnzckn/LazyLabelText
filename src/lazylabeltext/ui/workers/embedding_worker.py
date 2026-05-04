"""Embedding worker: batch embedding computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from lazylabeltext.core.protocols import EmbeddingProviderProtocol


class EmbeddingWorker(QThread):
    """Background thread for computing embeddings."""

    progress = pyqtSignal(int, int)  # current_batch, total_batches
    finished = pyqtSignal(object)  # np.ndarray
    error = pyqtSignal(str)

    BATCH_SIZE = 32

    def __init__(
        self,
        provider: EmbeddingProviderProtocol,
        texts: list[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.provider = provider
        self.texts = texts

    def run(self) -> None:
        try:
            all_embeddings = []
            total_batches = (len(self.texts) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

            for i in range(0, len(self.texts), self.BATCH_SIZE):
                batch = self.texts[i : i + self.BATCH_SIZE]
                embeddings = self.provider.encode(batch)
                all_embeddings.append(embeddings)
                self.progress.emit(i // self.BATCH_SIZE + 1, total_batches)

            result = np.vstack(all_embeddings) if all_embeddings else np.array([])
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
