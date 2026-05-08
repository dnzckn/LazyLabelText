"""Domain model dataclasses for LazyLabelText."""

from __future__ import annotations

from dataclasses import dataclass, field

# Reserved category name for "the LLM looked at this chunk and judged it
# doesn't fit any rubric category". A real, user-visible value (not just
# absence) so the timeline can render it distinctly from "pending /
# unlabeled" and analytics can count it as a deliberate outcome.
# Empty predicted_categories on a Label is treated as DNB at display time
# for backward compatibility with rows ingested before this constant existed.
DNB_CATEGORY = "DNB (does not belong)"


@dataclass
class Heading:
    """A heading found in a document."""

    level: int
    text: str
    char_start: int
    char_end: int


@dataclass
class PageBoundary:
    """A page boundary in a PDF document."""

    page_number: int
    char_start: int
    char_end: int


@dataclass
class SectionBoundary:
    """A section boundary defined by heading hierarchy."""

    section_path: list[str]
    char_start: int
    char_end: int


@dataclass
class ConvertedDocument:
    """A document converted to internal representation."""

    id: int | None = None
    filename: str = ""
    format: str = ""
    full_text: str = ""
    headings: list[Heading] = field(default_factory=list)
    pages: list[PageBoundary] = field(default_factory=list)
    sections: list[SectionBoundary] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    status: str = "pending"  # pending / parsed / warnings / failed
    warnings: list[str] = field(default_factory=list)
    ingested_at: str = ""
    # SHA256 of the source bytes — the canonical identity of a document.
    # Two ingest paths pointing at the same content produce the same hash,
    # so moves/copies are deduped without losing chunks/labels. Empty for
    # legacy rows ingested before content-hash dedupe; treat as unknown.
    source_hash: str = ""


@dataclass
class Category:
    """A labeling category within a rubric."""

    name: str = ""
    definition: str = ""
    exemplars: list[str] = field(default_factory=list)
    boundary_cases: list[str] = field(default_factory=list)
    confidence_threshold: float = 0.85
    parent: str | None = None


@dataclass
class Rubric:
    """A versioned labeling rubric."""

    id: int | None = None
    name: str = ""
    version: int = 1
    categories: list[Category] = field(default_factory=list)
    created_at: str = ""


@dataclass
class ChunkingRun:
    """A record of a chunking operation."""

    id: int | None = None
    document_id: int = 0
    strategy: str = ""
    params: dict = field(default_factory=dict)
    started_at: str = ""
    n_chunks: int = 0


@dataclass
class Chunk:
    """An atomic text chunk with provenance."""

    id: int | None = None
    document_id: int = 0
    chunking_run_id: int = 0
    text: str = ""
    char_start: int = 0
    char_end: int = 0
    section_path: list[str] = field(default_factory=list)
    token_count: int = 0
    chunk_type: str | None = None
    boundary_confidence: float | None = None
    manual_override: dict | None = None
    embedding: list[float] | None = None
    embedding_model: str | None = None


@dataclass
class ClassificationResult:
    """Result from an LLM classification call."""

    categories: list[str] = field(default_factory=list)
    confidence_per_category: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    # Mean log-probability of the model's response tokens, when the provider
    # exposes it (OpenAI, Google). Anthropic + Ollama leave this None.
    avg_logprob: float | None = None


@dataclass
class Label:
    """A label assigned to a chunk."""

    id: int | None = None
    chunk_id: int = 0
    rubric_version_id: int = 0
    predicted_categories: list[str] = field(default_factory=list)
    confidence_per_category: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    knn_agreement: float | None = None
    logprob_signal: float | None = None
    composite_confidence: float = 0.0
    llm_model: str = ""
    llm_run_id: str = ""
    created_at: str = ""


@dataclass
class HumanReview:
    """A human review of an LLM-proposed label."""

    id: int | None = None
    label_id: int = 0
    reviewer: str = "default"
    action: str = ""  # accept / correct / skip / flag
    final_categories: list[str] = field(default_factory=list)
    notes: str = ""
    reviewed_at: str = ""


@dataclass
class AuditEvent:
    """An immutable audit trail entry."""

    id: int | None = None
    event_type: str = ""
    actor: str = "system"
    timestamp: str = ""
    payload: dict = field(default_factory=dict)
    related_chunk_ids: list[int] = field(default_factory=list)
