"""JSON corpus exporter."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from lazylabeltext import __version__
from lazylabeltext.core.database import Database
from lazylabeltext.core.exporters import ExportFormat, _register

logger = logging.getLogger("lazylabeltext")


class JSONExporter:
    """Exports labeled corpus as a single JSON file with manifest."""

    def export(self, db: Database, rubric_version_id: int, output_path: str) -> str:
        """Export the labeled corpus to JSON."""
        rubric = db.get_rubric(rubric_version_id)
        if rubric is None:
            raise ValueError(f"Rubric version {rubric_version_id} not found")

        # Build chunks with labels and reviews
        all_labels = db.get_all_labels(rubric_version_id)
        chunks_data = []

        for label in all_labels:
            chunk = db.get_chunk(label.chunk_id)
            if chunk is None:
                continue

            doc = db.get_document(chunk.document_id)
            reviews = db.get_reviews_for_label(label.id or 0)

            chunk_entry = {
                "id": chunk.id,
                "text": chunk.text,
                "source": {
                    "document": doc.filename if doc else "unknown",
                    "section_path": chunk.section_path,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                },
                "token_count": chunk.token_count,
                "chunk_type": chunk.chunk_type,
                "boundary_confidence": chunk.boundary_confidence,
                "manual_override": chunk.manual_override,
                "label": {
                    "rubric_version": rubric.version,
                    "categories": label.predicted_categories,
                    "confidence": label.confidence_per_category,
                    "rationale": label.rationale,
                    "composite_confidence": label.composite_confidence,
                    "knn_agreement": label.knn_agreement,
                    "llm_model": label.llm_model,
                },
            }

            if reviews:
                review = reviews[-1]  # Latest review
                chunk_entry["label"]["human_review"] = {
                    "action": review.action,
                    "final_categories": review.final_categories,
                    "reviewer": review.reviewer,
                    "notes": review.notes,
                    "reviewed_at": review.reviewed_at,
                }

            chunks_data.append(chunk_entry)

        # Build manifest
        summary = db.get_labeling_summary(rubric_version_id)

        output = {
            "manifest": {
                "rubric_version": rubric.version,
                "rubric_name": rubric.name,
                "total_chunks": len(chunks_data),
                "labeled_chunks": summary["total_labels"],
                "reviewed_chunks": summary["reviewed"],
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "tool_version": __version__,
            },
            "rubric": {
                "name": rubric.name,
                "version": rubric.version,
                "categories": [
                    {
                        "name": c.name,
                        "definition": c.definition,
                        "exemplars": c.exemplars,
                    }
                    for c in rubric.categories
                ],
            },
            "chunks": chunks_data,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info("Exported %d chunks to %s", len(chunks_data), output_path)
        return output_path


_register(ExportFormat.JSON, JSONExporter())
