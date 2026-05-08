"""Label management: LLM classification and human review."""

from __future__ import annotations

import logging
import threading
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
        # Guards _exemplar_embeddings build/read so concurrent label calls
        # (parallel mode) don't both rebuild the cache and clobber it.
        self._exemplar_lock = threading.Lock()

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

        # Compute and persist the chunk embedding once; reuse it for kNN below
        # and store it on the chunk row for downstream RAG / export.
        chunk_embedding = None
        embedding_model_name = None
        if self.embedding_provider is not None:
            try:
                chunk_embedding = self.embedding_provider.encode_one(chunk.text)
                embedding_model_name = getattr(
                    self.embedding_provider, "model_name", "unknown"
                )
                if chunk.id is not None:
                    self.db.set_chunk_embedding(
                        chunk.id, chunk_embedding, embedding_model_name
                    )
                    chunk.embedding = (
                        chunk_embedding.tolist()
                        if hasattr(chunk_embedding, "tolist")
                        else list(chunk_embedding)
                    )
                    chunk.embedding_model = embedding_model_name
            except Exception:
                logger.debug("Embedding compute/persist failed", exc_info=True)
                chunk_embedding = None

        # kNN agreement reuses the same embedding (no second API call).
        knn_agreement = None
        if (
            self.embedding_provider is not None
            and rubric.categories
            and chunk_embedding is not None
        ):
            try:
                knn_agreement = self._compute_knn_agreement_with_embedding(
                    chunk_embedding, result.categories, rubric
                )
            except Exception:
                logger.debug("kNN agreement computation failed", exc_info=True)

        composite = self._compute_composite_confidence(
            result.confidence_per_category, knn_agreement, result.avg_logprob
        )

        label = Label(
            chunk_id=chunk.id or 0,
            rubric_version_id=rubric.id or 0,
            predicted_categories=result.categories,
            confidence_per_category=result.confidence_per_category,
            rationale=result.rationale,
            knn_agreement=knn_agreement,
            logprob_signal=result.avg_logprob,
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

    def bypass_unreviewed(
        self,
        rubric_version_id: int,
        document_ids: list[int] | None = None,
    ) -> int:
        """Mark every unreviewed label as accepted-without-review.

        Records a HumanReview with action="bypassed" (distinct from
        "accept" so downstream consumers can tell deliberate human
        approvals from bulk pass-throughs). Existing reviews are left
        alone — bypass never overrides manual review work. Returns the
        count of labels that got a new bypass review.

        ``document_ids`` scopes the bypass to specific docs; None means
        every doc that has labels for this rubric.
        """
        labels = self.db.get_all_labels(rubric_version_id)
        scope: set[int] | None = (
            set(document_ids) if document_ids is not None else None
        )
        bypassed = 0
        for lab in labels:
            if lab.id is None:
                continue
            existing = self.db.get_reviews_for_label(lab.id)
            if existing:
                continue
            if scope is not None:
                chunk = self.db.get_chunk(lab.chunk_id)
                if chunk is None or chunk.document_id not in scope:
                    continue
            self.submit_review(
                lab.id,
                action="bypassed",
                final_categories=list(lab.predicted_categories or []),
            )
            bypassed += 1
        return bypassed

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

    def drop_dnb_labels(
        self,
        rubric_version_id: int,
        document_id: int | None = None,
    ) -> int:
        """Delete all DNB ("does not belong") labels for a rubric.

        If ``document_id`` is given, only that doc's DNB labels are removed
        — the chunk goes back to "unlabeled" so it can be relabeled in a
        future round (e.g. with an updated rubric that adds a category
        that matches what the LLM previously rejected). Returns the count
        deleted. Empty predicted_categories are treated as DNB too — that
        covers legacy rows from before the DNB constant existed.
        """
        from lazylabeltext.core.models import DNB_CATEGORY

        labels = self.db.get_all_labels(rubric_version_id)
        deleted = 0
        for lab in labels:
            cats = lab.predicted_categories or []
            is_dnb = (not cats) or (
                len(cats) == 1 and cats[0] == DNB_CATEGORY
            )
            if not is_dnb:
                continue
            if document_id is not None:
                # Only drop DNBs for this doc.
                chunk = self.db.get_chunk(lab.chunk_id)
                if chunk is None or chunk.document_id != document_id:
                    continue
            if lab.id is not None:
                self.db.delete_label(lab.id)
                deleted += 1
        return deleted

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
        """Compute kNN agreement starting from raw text (computes its own embedding)."""
        if not self.embedding_provider:
            return 0.0
        chunk_emb = self.embedding_provider.encode_one(chunk_text)
        return self._compute_knn_agreement_with_embedding(
            chunk_emb, predicted, rubric
        )

    def _compute_knn_agreement_with_embedding(
        self, chunk_emb, predicted: list[str], rubric: Rubric
    ) -> float:
        """kNN agreement when the chunk embedding is already computed."""
        if not self.embedding_provider:
            return 0.0

        # Build-once-then-read: lock the build so concurrent threads share
        # one rebuild, then snapshot the dict so the dot-product loop runs
        # outside the lock (cache invalidation only happens via
        # set_embedding_provider, which assigns None).
        with self._exemplar_lock:
            if self._exemplar_embeddings is None:
                self._build_exemplar_embeddings(rubric)
            cache = self._exemplar_embeddings

        if not cache:
            return 0.0

        best_category = ""
        best_sim = -1.0
        for cat_name, emb in cache.items():
            sim = float(
                np.dot(chunk_emb, emb)
                / (np.linalg.norm(chunk_emb) * np.linalg.norm(emb) + 1e-8)
            )
            if sim > best_sim:
                best_sim = sim
                best_category = cat_name

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
        avg_logprob: float | None = None,
    ) -> float:
        """Compute composite confidence from available signals.

        Sources, in order of trust:
          - LLM self-reported confidence (always available, weak)
          - kNN exemplar agreement (when embeddings configured)
          - avg token log-probability (when the provider exposes it)

        Each signal is folded in if present and silently skipped if not, so
        Anthropic + Ollama (no logprobs) and projects without an embedding
        provider both still get a sensible composite.
        """
        if not confidence_per_category:
            return 0.0

        llm_confidence = max(confidence_per_category.values())
        composite = llm_confidence

        if knn_agreement is not None:
            if llm_confidence > 0.8 and knn_agreement > 0.5:
                composite = min(1.0, llm_confidence * 1.05)
            elif llm_confidence > 0.8 and knn_agreement == 0.0:
                composite = llm_confidence * 0.7
            else:
                composite = llm_confidence * 0.85

        if avg_logprob is not None:
            # avg_logprob in (-inf, 0]; > -0.5 ≈ very confident, < -1.5 ≈ uncertain.
            if avg_logprob > -0.5:
                composite = min(1.0, composite * 1.03)
            elif avg_logprob < -1.5:
                composite = composite * 0.92

        return max(0.0, min(1.0, composite))
