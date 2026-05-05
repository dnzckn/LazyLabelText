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

        classify_text = self._build_classification_text(chunk)
        try:
            result = self.llm_provider.classify(classify_text, rubric.categories)
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
        self,
        rubric_version_id: int,
        threshold: float = 1.0,
        document_id: int | None = None,
    ) -> list[tuple[Chunk, Label]]:
        """Get chunks needing review, sorted by confidence ascending."""
        return self.db.get_chunks_for_review(
            rubric_version_id, threshold, document_id
        )

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

    def get_unlabeled_chunks(
        self, rubric_version_id: int, document_id: int | None = None
    ) -> list[Chunk]:
        return self.db.get_unlabeled_chunks(rubric_version_id, document_id)

    def _build_classification_text(self, chunk: Chunk) -> str:
        """Compose chunk text with section_path context for classification.

        The provider only ever sees a string; section_path stays out of the
        provider contract and out of stored chunk.text.
        """
        if not chunk.section_path:
            return chunk.text
        section = " > ".join(chunk.section_path)
        return f"[Section: {section}]\n\n{chunk.text}"

    def delete_label(self, label_id: int) -> None:
        """Discard a single label so its chunk re-enters the unlabeled pool."""
        self.db.delete_label(label_id)

    def clear_labels(self, rubric_version_id: int) -> int:
        """Delete all labels for a rubric version. Returns the number deleted."""
        return self.db.delete_labels_for_rubric(rubric_version_id)

    def clear_labels_for_document(self, document_id: int) -> int:
        """Delete all labels (any rubric version) for a single document."""
        return self.db.delete_labels_for_document(document_id)

    def get_category_coverage(self, rubric_version_id: int) -> list[dict]:
        """Per-category corpus health stats for the rubric coverage view.

        Returns one dict per category in the rubric:
            name, count, avg_confidence, doc_count, disagree_pct, review_count
        """
        rubric = self.db.get_rubric(rubric_version_id)
        if rubric is None:
            return []

        labels = self.db.get_all_labels(rubric_version_id)

        stats: dict[str, dict] = {}

        def _slot(name: str) -> dict:
            if name not in stats:
                stats[name] = {
                    "count": 0,
                    "confidence_sum": 0.0,
                    "doc_ids": set(),
                    "review_count": 0,
                    "correct_count": 0,
                }
            return stats[name]

        for cat in rubric.categories:
            _slot(cat.name)

        for label in labels:
            chunk = self.db.get_chunk(label.chunk_id)
            if chunk is None:
                continue
            reviews = self.db.get_reviews_for_label(label.id or 0)
            review = reviews[-1] if reviews else None
            for cat_name in label.predicted_categories or []:
                slot = _slot(cat_name)
                slot["count"] += 1
                slot["confidence_sum"] += label.composite_confidence or 0.0
                slot["doc_ids"].add(chunk.document_id)
                if review:
                    slot["review_count"] += 1
                    if review.action == "correct":
                        slot["correct_count"] += 1

        result: list[dict] = []
        seen = set()

        def _build_entry(name: str, in_rubric: bool) -> dict:
            s = stats.get(name) or {
                "count": 0,
                "confidence_sum": 0.0,
                "doc_ids": set(),
                "review_count": 0,
                "correct_count": 0,
            }
            count = s["count"]
            return {
                "name": name,
                "in_rubric": in_rubric,
                "count": count,
                "avg_confidence": s["confidence_sum"] / count if count else 0.0,
                "doc_count": len(s["doc_ids"]),
                "review_count": s["review_count"],
                "disagree_pct": (
                    s["correct_count"] / s["review_count"] if s["review_count"] else 0.0
                ),
            }

        for cat in rubric.categories:
            result.append(_build_entry(cat.name, in_rubric=True))
            seen.add(cat.name)

        # Stragglers: labels whose category isn't in the current rubric
        # (e.g. labeled under a previous version). Surface them so the user notices.
        for name in stats:
            if name not in seen:
                result.append(_build_entry(name, in_rubric=False))

        return result

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
