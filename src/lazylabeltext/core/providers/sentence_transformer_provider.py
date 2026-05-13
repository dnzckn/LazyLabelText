"""Sentence-transformers embedding provider with local model cache.

Loads each model from a local directory under `Paths().models_dir` if
present; otherwise downloads once via the standard SentenceTransformer
constructor and saves to that directory so subsequent runs skip the
network call.

Pre-populating the cache without running the app:

    pip install sentence-transformers
    python -c "from sentence_transformers import SentenceTransformer; \
                SentenceTransformer('all-MiniLM-L6-v2').save( \
                '<lazylabeltext>/models/all-MiniLM-L6-v2')"
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from lazylabeltext.config.paths import Paths
from lazylabeltext.core.exceptions import EmbeddingProviderError

logger = logging.getLogger("lazylabeltext")


class SentenceTransformerProvider:
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        # Concurrent encode_one calls (parallel labeling) all hit _get_model
        # together. Without a lock, every thread sees _model=None on the
        # first call and each independently logs "Downloading..." and runs
        # the SentenceTransformer constructor — wasted work and log spam.
        # Double-checked locking: cheap fast-path once the model is loaded.
        self._model_lock = threading.Lock()
        # Belt and suspenders: even if the ctor raises and gets retried, only
        # log the "Downloading…" notice once per provider instance.
        self._announced_download = False
        # SentenceTransformer.encode is *not* documented thread-safe under
        # heavy concurrency. Tokenizer caches and PyTorch internal buffers
        # can race when several workers call encode at once. A serialized
        # encode adds <50ms per chunk, which is rounding error vs the
        # 1–3 s LLM call and totally worth the safety margin.
        self._encode_lock = threading.Lock()

    def _local_path(self):
        return Paths().model_path(self.model_name)

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
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

            # 1. Local cache hit → load from disk, no network call.
            if local_path.exists() and any(local_path.iterdir()):
                try:
                    logger.info(
                        "Loading cached embedding model from %s", local_path
                    )
                    self._model = SentenceTransformer(str(local_path))
                    return self._model
                except Exception as e:
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

            # 2. Download via the SentenceTransformer constructor (uses HF
            #    Hub cache under the hood). Save a copy to our local
            #    models_dir so subsequent runs load from disk.
            if not self._announced_download:
                logger.info(
                    "Downloading embedding model '%s' (one-time)...",
                    self.model_name,
                )
                self._announced_download = True
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
        with self._encode_lock:
            return model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True,
            )

    def encode_one(self, text: str) -> np.ndarray:
        result = self.encode([text])
        return result[0]

    def warm_up(self) -> None:
        """Force the model to load + first-call init on the calling thread.

        Used by the parallel orchestrator before spawning workers so the
        lazy SentenceTransformer load doesn't bottleneck N-1 of N workers
        on the first chunk. Safe to call repeatedly.
        """
        self._get_model()
        # Run a tiny encode to flush any first-call CUDA / kernel JIT.
        with self._encode_lock:
            self._model.encode(
                ["warmup"], show_progress_bar=False, convert_to_numpy=True,
            )

    def is_loaded(self) -> bool:
        return self._model is not None
