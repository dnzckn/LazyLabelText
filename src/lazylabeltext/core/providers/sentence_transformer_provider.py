"""Sentence-transformers embedding provider with local model cache.

Mirrors the LazyLabel pattern: each model is loaded from a local directory
under `Paths().models_dir` if present; otherwise it's downloaded once via
the standard SentenceTransformer constructor and then saved to that
directory for future offline use.

Manual install path (no internet on target machine):

    On a machine with internet:
      pip install sentence-transformers
      python -c "from sentence_transformers import SentenceTransformer; \
                  SentenceTransformer('all-MiniLM-L6-v2').save( \
                  '<lazylabeltext>/models/all-MiniLM-L6-v2')"

    Then copy the resulting folder to the offline machine's
    `<lazylabeltext-install>/models/` directory.
"""

from __future__ import annotations

import logging

import numpy as np

from lazylabeltext.config.paths import Paths
from lazylabeltext.core.exceptions import EmbeddingProviderError

logger = logging.getLogger("lazylabeltext")


class SentenceTransformerProvider:
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _local_path(self):
        return Paths().model_path(self.model_name)

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingProviderError(
                "sentence-transformers",
                "sentence-transformers package not installed",
            ) from e

        local_path = self._local_path()

        # 1. Local cache hit → load offline.
        if local_path.exists() and any(local_path.iterdir()):
            try:
                logger.info(
                    "Loading cached embedding model from %s", local_path
                )
                self._model = SentenceTransformer(str(local_path))
                return self._model
            except Exception as e:
                # Corrupt cache → wipe + fall through to re-download.
                logger.warning(
                    "Cached model at %s failed to load (%s); re-downloading.",
                    local_path,
                    e,
                )
                try:
                    import shutil

                    shutil.rmtree(local_path, ignore_errors=True)
                except Exception:
                    pass

        # 2. Download via the SentenceTransformer constructor (uses HF Hub
        #    cache under the hood). Then save a copy to our local models_dir
        #    so subsequent runs — including offline runs — load from disk.
        logger.info(
            "Downloading embedding model '%s' (one-time)...", self.model_name
        )
        try:
            model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise EmbeddingProviderError(
                "sentence-transformers",
                (
                    f"Failed to download '{self.model_name}': {e}\n\n"
                    "To install manually, run on a machine with internet:\n"
                    f"  python -c \"from sentence_transformers import "
                    f"SentenceTransformer; "
                    f"SentenceTransformer('{self.model_name}').save("
                    f"'{local_path}')\"\n"
                    f"Then copy the resulting folder to:\n  {local_path}"
                ),
            ) from e

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(local_path))
            logger.info("Cached embedding model to %s", local_path)
        except Exception as e:
            # Saving locally is a nicety; runtime still works via HF cache.
            logger.warning("Could not save model to local cache: %s", e)

        self._model = model
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def encode_one(self, text: str) -> np.ndarray:
        result = self.encode([text])
        return result[0]

    def is_loaded(self) -> bool:
        return self._model is not None
