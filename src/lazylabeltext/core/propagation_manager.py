"""Propagation manager: batch apply chunking + labeling across corpus."""

from __future__ import annotations

import logging
from collections.abc import Callable

from lazylabeltext.core.chunk_manager import ChunkManager
from lazylabeltext.core.database import Database
from lazylabeltext.core.document_manager import DocumentManager
from lazylabeltext.core.label_manager import LabelManager
from lazylabeltext.core.models import Rubric

logger = logging.getLogger("lazylabeltext")


class PropagationManager:
    """Batch processes all documents with current chunking + labeling settings."""

    def __init__(
        self,
        database: Database,
        document_manager: DocumentManager,
        chunk_manager: ChunkManager,
        label_manager: LabelManager,
    ) -> None:
        self.db = database
        self.document_manager = document_manager
        self.chunk_manager = chunk_manager
        self.label_manager = label_manager

    def propagate_all(
        self,
        chunking_strategy: str,
        chunking_params: dict,
        rubric: Rubric,
        progress_callback: Callable[[int, int, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict:
        """Apply chunking and labeling to all documents.

        Args:
            progress_callback: Called with (current_doc, total_docs, status_message)
            should_stop: Called to check if operation should be cancelled

        Returns:
            Summary dict with counts and any errors.
        """
        documents = self.document_manager.get_all_documents()
        total = len(documents)
        results = {
            "total": total,
            "processed": 0,
            "chunks_created": 0,
            "labels_created": 0,
            "errors": [],
        }

        for i, doc in enumerate(documents):
            if should_stop and should_stop():
                logger.info("Propagation cancelled by user")
                break

            if doc.status == "failed":
                results["errors"].append(
                    {"document": doc.filename, "error": "Document failed to parse"}
                )
                continue

            if progress_callback:
                progress_callback(i + 1, total, f"Processing {doc.filename}...")

            try:
                # Chunk the document
                run = self.chunk_manager.run_chunking(
                    doc.id,
                    chunking_strategy,
                    chunking_params,  # type: ignore[arg-type]
                )
                chunks = self.chunk_manager.get_chunks(
                    doc.id,
                    run.id,  # type: ignore[arg-type]
                )
                results["chunks_created"] += len(chunks)

                # Label all chunks
                if self.label_manager.llm_provider is not None:
                    labels = self.label_manager.label_batch(chunks, rubric)
                    results["labels_created"] += len(labels)

                results["processed"] += 1

            except Exception as e:
                logger.error("Propagation error on %s: %s", doc.filename, e)
                results["errors"].append({"document": doc.filename, "error": str(e)})

        logger.info(
            "Propagation complete: %d/%d docs, %d chunks, %d labels",
            results["processed"],
            total,
            results["chunks_created"],
            results["labels_created"],
        )
        return results
