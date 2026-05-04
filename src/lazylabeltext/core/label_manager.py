"""Label management: LLM classification and human review."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable

import numpy as np

from lazylabeltext.core.database import Database
from lazylabeltext.core.exceptions import (
    ClassificationError,
    ProviderNotConfiguredError,
)
from lazylabeltext.core.models import Chunk, HumanReview, Label, Rubric
from lazylabeltext.core.protocols import EmbeddingProviderProtocol, LLMProviderProtocol

logger = logging.getLogger("lazylabeltext")


class LabelManager:
    """Orchestrates LLM labeling and human review."""

    def __init__(
        self,
        database: Database,
        llm_provider: LLMProviderProtocol | None = None,
        embedding_provider: EmbeddingProviderProtocol | None = None,
    ) -> None:
        self.db = database
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self._exemplar_embeddings: dict[str, np.ndarray] | None = None

    def set_llm_provider(self, provider: LLMProviderProtocol | None) -> None:
        self.llm_provider = provider

    def set_embedding_provider(
        self, provider: EmbeddingProviderProtocol | None
    ) -> None:
        self.embedding_provider = provider
        self._exemplar_embeddings = None

    def label_chunk(self, chunk: Chunk, rubric: Rubric) -> Label:
        """Label a single chunk using the LLM and optional kNN."""
        if self.llm_provider is None:
            raise ProviderNotConfiguredError("LLM")

        run_id = str(uuid.uuid4())[:8]

        try:
            result = self.llm_provider.classify(chunk.text, rubric.categories)
        except Exception as e:
            raise ClassificationError(chunk.id or 0, str(e)) from e

        # Compute kNN agreement if embeddings available
        knn_agreement = None
        if self.embedding_provider is not None and rubric.categories:
            try:
                knn_agreement = self._compute_knn_agreement(
                    chunk.text, result.categories, rubric
                )
            except Exception:
                logger.debug("kNN agreement computation failed", exc_info=True)

        composite = self._compute_composite_confidence(
            result.confidence_per_category, knn_agreement
        )

        label = Label(
            chunk_id=chunk.id or 0,
            rubric_version_id=rubric.id or 0,
            predicted_categories=result.categories,
            confidence_per_category=result.confidence_per_category,
            rationale=result.rationale,
            knn_agreement=knn_agreement,
            composite_confidence=composite,
            llm_model=getattr(self.llm_provider, "model", "unknown"),
            llm_run_id=run_id,
        )
        label.id = self.db.insert_label(label)
        return label

    def label_batch(
        self,
        chunks: list[Chunk],
        rubric: Rubric,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Label]:
        """Label a batch of chunks."""
        labels: list[Label] = []
        for i, chunk in enumerate(chunks):
            label = self.label_chunk(chunk, rubric)
            labels.append(label)
            if progress_callback:
                progress_callback(i + 1, len(chunks))
        return labels

    def get_review_queue(
        self, rubric_version_id: int, threshold: float = 1.0
    ) -> list[tuple[Chunk, Label]]:
        """Get chunks needing review, sorted by confidence ascending."""
        return self.db.get_chunks_for_review(rubric_version_id, threshold)

    def submit_review(
        self,
        label_id: int,
        action: str,
        final_categories: list[str] | None = None,
        notes: str = "",
        reviewer: str = "default",
    ) -> HumanReview:
        """Record a human review decision."""
        review = HumanReview(
            label_id=label_id,
            reviewer=reviewer,
            action=action,
            final_categories=final_categories or [],
            notes=notes,
        )
        review.id = self.db.insert_review(review)
        return review

    def get_labeling_summary(self, rubric_version_id: int) -> dict:
        return self.db.get_labeling_summary(rubric_version_id)

    def get_unlabeled_chunks(self, rubric_version_id: int) -> list[Chunk]:
        return self.db.get_unlabeled_chunks(rubric_version_id)

    def _compute_knn_agreement(
        self, chunk_text: str, predicted: list[str], rubric: Rubric
    ) -> float:
        """Compute agreement between LLM prediction and kNN from exemplars."""
        if not self.embedding_provider:
            return 0.0

        # Build or retrieve exemplar embeddings
        if self._exemplar_embeddings is None:
            self._build_exemplar_embeddings(rubric)

        if not self._exemplar_embeddings:
            return 0.0

        chunk_emb = self.embedding_provider.encode_one(chunk_text)

        # Find nearest exemplar category
        best_category = ""
        best_sim = -1.0
        for cat_name, emb in self._exemplar_embeddings.items():
            sim = float(
                np.dot(chunk_emb, emb)
                / (np.linalg.norm(chunk_emb) * np.linalg.norm(emb) + 1e-8)
            )
            if sim > best_sim:
                best_sim = sim
                best_category = cat_name

        # Agreement: 1.0 if kNN agrees with prediction, 0.0 if not
        if best_category in predicted:
            return 1.0
        return 0.0

    def _build_exemplar_embeddings(self, rubric: Rubric) -> None:
        """Build mean embeddings from category exemplars."""
        self._exemplar_embeddings = {}
        if not self.embedding_provider:
            return

        for cat in rubric.categories:
            if cat.exemplars:
                embs = self.embedding_provider.encode(cat.exemplars)
                self._exemplar_embeddings[cat.name] = np.mean(embs, axis=0)

    def _compute_composite_confidence(
        self,
        confidence_per_category: dict[str, float],
        knn_agreement: float | None,
    ) -> float:
        """Compute composite confidence from available signals."""
        if not confidence_per_category:
            return 0.0

        llm_confidence = max(confidence_per_category.values())

        if knn_agreement is not None:
            if llm_confidence > 0.8 and knn_agreement > 0.5:
                return min(1.0, llm_confidence * 1.05)  # Boost
            elif llm_confidence > 0.8 and knn_agreement == 0.0:
                return llm_confidence * 0.7  # Penalize disagreement
            else:
                return llm_confidence * 0.85

        return llm_confidence
