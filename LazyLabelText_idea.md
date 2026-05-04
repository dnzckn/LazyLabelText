# LazyLabelText

A focused desktop tool for producing high-quality labeled text corpora from document collections. Sister project to LazyLabel (image segmentation labeling) — same philosophy, different modality.

## What it is

LazyLabelText takes a folder of source documents (PDF, Word, Markdown, plain text) and helps a human + LLM team produce a labeled corpus of atomic chunks. Each chunk carries provenance, a label assignment from a tunable rubric, and a confidence signal. The output is a structured artifact (JSON, JSONL, or Parquet) ready to feed downstream systems: consolidation pipelines, RAG indexes, training data, knowledge graphs, audit corpora.

The tool's job ends at "labeled corpus." It does not consolidate, synthesize, or generate. Those are downstream concerns and belong in separate tools.

## What it isn't

- Not a consolidation engine. It produces the input to consolidation, not consolidated output.
- Not a RAG. It produces the input to RAG, not retrieval or generation.
- Not a training pipeline. It produces the input to training.
- Not an annotation tool for unstructured tagging. The rubric is structured and central.
- Not a SaaS. Local-first, runs against local files. Cloud LLM calls are configurable.

## Core principles

1. **Rubric is the artifact.** The labeling rubric (what categories exist, what each means, what exemplars belong to each) is treated as a first-class versioned artifact, not a config file. Rubrics get iterated against real corpus content and saved with the labels they produced.

2. **Chunking is tunable and inspectable.** Users can choose chunking strategy (structural, semantic, LLM, hybrid), see the resulting chunks immediately, adjust thresholds, and re-chunk without re-loading. Bad chunks produce bad labels; chunking quality is visible at every step.

3. **Humans confirm, LLM proposes.** The LLM produces a labeling proposal with confidence. The human accepts, corrects, or escalates. Every disagreement is captured as training signal for the next rubric revision.

4. **Provenance is non-negotiable.** Every chunk carries source doc, page, section, char offset. Every label change is logged with timestamp, actor, and prior value. The audit trail is queryable.

5. **Local-first.** Documents don't leave the user's machine. LLM calls are pluggable (local model via Ollama, hosted API via Anthropic/OpenAI). Embeddings can run locally via sentence-transformers.

6. **Lazy in name, deliberate in design.** "Lazy" is the user experience — minimum clicks, smart defaults, fast iteration. The tool itself is opinionated and structured.

## Core workflows

### Workflow 1: Load and convert

User points the tool at a folder. It walks the folder, identifies file types, and converts each to plain text with structural metadata preserved.

Supported formats:
- PDF (text-based and OCR for scanned)
- DOCX (preserves heading styles)
- HTML
- Markdown
- Plain text
- RTF

Output of this stage: a normalized internal representation per document, preserving:
- The full text
- Heading hierarchy with levels
- Page boundaries (PDF) or section boundaries (DOCX)
- Original character offsets for citation
- Document metadata (author, dates, source filename)

User sees a document list with status: parsed / has warnings / failed. Clicking a document shows the converted text alongside a thumbnail of the original.

### Workflow 2: Define and tune the rubric

The rubric is a structured definition of:
- **Categories** (the labels). Flat or hierarchical. Multi-label allowed.
- **Definitions.** What each category means in plain language.
- **Exemplars.** 2-10 example chunks per category that ground the LLM and serve as kNN reference points.
- **Boundary cases.** Optional notes on common edge cases and how to resolve them.
- **Confidence rules.** Per-category thresholds for auto-accept vs. flag for review.

Users can:
- Create a rubric from scratch
- Import from JSON
- Bootstrap from existing labeled data (cluster the data, name the clusters, those become categories)
- Iterate by viewing classifier mistakes and updating definitions/exemplars
- Version the rubric — every revision is saved and labels track which rubric version produced them

The rubric editor is a side-by-side view: rubric definitions on the left, sample chunks from the corpus on the right. The user can drag chunks into category slots as exemplars without leaving the editor.

### Workflow 3: Chunk and inspect

The user picks a chunking strategy and parameters. The tool chunks all loaded documents and shows the result.

Available strategies:

1. **Structural.** Parse headings, numbered sections, list items, definition blocks. Fast, deterministic, free. Good first pass.
2. **Semantic.** Sentence-level embeddings, rolling cosine similarity, place boundaries at similarity drops. Good for unstructured prose.
3. **LLM-based.** Send sections to a small LLM with a "split into atomic clauses" prompt. Slow, expensive, most accurate.
4. **Hybrid (recommended default).** Structural pass first, length/shape check, semantic pass on long chunks, LLM pass on residual ambiguous chunks.
5. **Custom.** User provides a Python function or regex.

Tunable parameters per strategy:
- Token thresholds (min, target, max chunk size)
- Semantic similarity threshold for boundary placement
- Window size for semantic chunking (1, 2, or 3 sentences)
- Heading levels to split on
- LLM model and prompt for the LLM strategy

The chunks panel shows:
- Each chunk's text
- Token count
- Source location (document, page, section, char range)
- Detected chunk_type (procedure / principle / part_line / etc., per the rubric or pluggable inferrer)
- Boundary confidence (from dual-run agreement, when enabled)
- A "merge with neighbor" / "split here" affordance for manual override

The user can inspect chunk quality at a glance: histogram of token counts, fraction that look atomic (heuristic), number flagged for boundary uncertainty. They can re-chunk with different parameters and compare results side-by-side. The tool does not commit chunks until the user explicitly says "use these."

Manual overrides (merge / split / re-text) are recorded as part of the chunking output, so they survive re-runs and can be replayed.

### Workflow 4: Run the LLM, review, label

With chunks committed and rubric loaded, the user runs the labeler.

Per chunk, the labeler produces:
- Predicted category (or categories, for multi-label)
- Confidence per category
- Rationale (short text from the LLM explaining the choice)
- Embedding-kNN agreement score against the rubric's exemplars
- Self-consistency score (if multi-run is enabled)
- Composite confidence (calibrated combination of the above)

The review UI presents chunks one at a time, sorted by composite confidence ascending (lowest first — humans see ambiguous cases when fresh). Each chunk shows:
- Chunk text in the center
- Source citation underneath
- Predicted labels with confidence bars
- Rationale from the LLM
- Nearest exemplars from the rubric for visual comparison
- Quick keys: 1-9 to accept the proposed label(s), letters to switch to alternative categories, `e` to edit chunk text, `s` to skip, `f` to flag for SME, `n` to add a note

Auto-accept threshold: chunks above the configurable confidence threshold are auto-labeled, with the user reviewing only the ambiguous tail. The threshold is per-category and tunable.

After review, the user sees a labeling summary: counts per category, distribution of confidence, number of disagreements between LLM and human, top categories by review time. This drives the next rubric revision.

### Workflow 5: Export

The labeled corpus exports to:
- **JSON / JSONL.** One chunk per record, full provenance, all confidence signals, label history.
- **Parquet.** Columnar, suitable for downstream ML.
- **CSV.** Flattened for spreadsheet review.
- **Custom Python.** User-provided post-processor — useful for adapting to a downstream tool's exact schema.

The export bundle includes the rubric version used and a manifest with corpus statistics, so downstream consumers can reproduce or audit.

## Architecture

### Stack

- **Frontend.** Electron + React, or Tauri + React for a smaller binary. Tauri is preferred for filesystem-heavy local apps.
- **Backend.** Python core (this is where the chunking, embedding, LLM logic lives). Communicates with the frontend over local IPC or HTTP.
- **Storage.** SQLite for the project database (chunks, labels, rubric versions, audit log). File system for source documents and exports.
- **LLM provider abstraction.** Pluggable: Anthropic, OpenAI, Google, Ollama (local), llama.cpp (local). One interface, swap per project.
- **Embedding provider abstraction.** Same shape. Default to sentence-transformers running locally; option to use hosted embedding APIs.

### Data model

```
Project
  ├─ documents (id, filename, format, ingested_at, status, metadata_json)
  ├─ rubric_versions (id, name, created_at, definitions_json, exemplars_json)
  ├─ chunking_runs (id, strategy, params_json, started_at, n_chunks)
  ├─ chunks (id, document_id, chunking_run_id, text, char_start, char_end,
  │           section_path, token_count, chunk_type, boundary_confidence,
  │           manual_override_json)
  ├─ labels (id, chunk_id, rubric_version_id, predicted_categories,
  │          confidence_per_category, rationale, knn_agreement,
  │          consistency_score, composite_confidence, llm_model, llm_run_id)
  ├─ human_reviews (id, label_id, reviewer, action, final_categories,
  │                 notes, reviewed_at)
  └─ audit_events (id, event_type, actor, timestamp, payload_json,
                    related_chunk_ids[])
```

Every state-changing operation writes to `audit_events` regardless of whether the user actively logs anything. Reproducibility falls out of this — given a project, you can reconstruct the labels at any prior moment.

### LLM and embedding provider interfaces

```python
class LLMProvider(Protocol):
    async def classify(
        self, chunk_text: str, rubric: Rubric
    ) -> ClassificationResult: ...

    async def chunk_text(
        self, text: str, prompt: str
    ) -> list[ChunkBoundary]: ...

class EmbeddingProvider(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...
    def encode_one(self, text: str) -> np.ndarray: ...
```

Concrete implementations:
- `AnthropicProvider`, `OpenAIProvider`, `GoogleProvider` (hosted)
- `OllamaProvider`, `LlamaCppProvider` (local)
- `SentenceTransformersProvider` (local embeddings, default)

### Confidence calibration

The composite confidence signal combines:
- LLM self-reported confidence (weakest)
- Logprobs on the predicted label token (when provider exposes them)
- Self-consistency across N runs at temperature 0.3-0.5
- Cosine similarity between chunk embedding and nearest rubric exemplar
- Agreement between LLM-chosen category and embedding-kNN-chosen category

Combination is via a simple decision tree:
- All signals agree, all confident → HIGH
- LLM confident but kNN disagrees → MEDIUM (review queue)
- LLM uncertain or signals split → LOW (review queue)

Per-project calibration: users can label a held-out 100-300 chunks themselves, the tool computes accuracy at each composite confidence band, and the auto-accept threshold gets tuned from that. This is the difference between "we have confidence numbers" and "we know what they mean."

## UI sketch

The main window is a three-pane layout:

```
┌─────────────┬──────────────────────────────┬──────────────────┐
│             │                              │                  │
│  Documents  │                              │     Rubric       │
│             │                              │                  │
│  ☑ doc1.pdf │     CHUNK / LABEL VIEW       │  Category: A     │
│  ☑ doc2.pdf │                              │  Definition...   │
│  ☐ doc3.md  │  (depends on current mode)   │  Exemplars: ...  │
│             │                              │                  │
│             │                              │  Category: B     │
│             │                              │  ...             │
│             │                              │                  │
│  + Add docs │                              │  + Edit rubric   │
│             │                              │                  │
└─────────────┴──────────────────────────────┴──────────────────┘
```

Top toolbar switches modes:
- **Convert** — see source vs. converted text per document
- **Rubric** — edit categories, definitions, exemplars
- **Chunk** — pick strategy, see chunks, tune
- **Label** — review LLM proposals, accept/correct
- **Export** — pick format, run export

Status bar at the bottom: project name, doc count, chunk count, label count, last LLM call time, last save.

The center pane is mode-specific:
- Convert mode: split view, original on left, converted on right.
- Rubric mode: rubric on top, sample chunks below — drag to add as exemplars.
- Chunk mode: chunked text with boundary markers, parameter panel on right side.
- Label mode: one chunk centered, hotkeys for fast classification.
- Export mode: format options, preview, save destination.

Keyboard-first design. Mouse usable but the labeling workflow especially should be drivable from the keyboard alone.

## Phasing

### Phase 1 — minimum viable version (4-6 weeks)

- Document loading: PDF (text only), DOCX, plain text, Markdown.
- Conversion to internal representation with heading hierarchy.
- Manual rubric definition (JSON file editor).
- Chunking: structural + manual merge/split overrides only.
- Labeling: single LLM provider (Anthropic), single embedding provider (sentence-transformers).
- Review UI: keyboard-driven label confirmation.
- Export: JSON only.

### Phase 2 — usable for real projects (8-10 weeks total)

- OCR for scanned PDFs.
- Semantic chunking with rolling cosine.
- Rubric editor UI with drag-to-exemplar.
- Multiple LLM providers.
- Confidence calibration workflow (held-out set, accuracy bands).
- JSONL and Parquet export.
- Project save/load.

### Phase 3 — production-grade (4-6 months total)

- LLM-based chunking option.
- Multi-user support (separate `reviewer` field already in schema; add auth).
- Bulk operations (relabel all chunks in category X, etc.).
- Diff view between rubric versions.
- Search across chunks and labels.
- Plugin API for custom chunkers, custom export formats.
- Telemetry on labeling time, agreement rates, rubric stability.

### Phase 4 — ecosystem fit (open-ended)

- Direct export connectors to common downstream tools.
- Integration with the consolidation pipeline (this tool's labeled output is that tool's input — no manual conversion).
- Shared rubric library (open source rubrics for common domains: contracts, requirements, BOMs, SOPs).
- CLI for scripted labeling runs.

## Open design questions

These are decisions worth deferring until the prototype reveals which way to go:

1. **Single project file vs. project folder.** A `.lltext` zipfile is portable but harder to inspect. A folder of files is git-friendly but less discrete. Probably folder, but worth testing both.

2. **Multi-document chunking vs. per-document.** Should chunks be addressed across the whole project or per-document? Per-document is simpler; cross-document is needed when chunks reference each other (cross-cite extraction). Default per-document, allow cross-document later.

3. **How aggressively to auto-accept.** A confidence threshold of 0.85 might auto-label 60% of a corpus, freeing humans for the hard 40%. Or it might quietly mislabel 5% of those auto-accepted. The right answer depends on calibration and the cost of errors. Start conservative (no auto-accept) and let users opt in.

4. **Rubric versioning model.** Linear versions (v1, v2, v3) or branched (different rubrics for different sub-corpora that may merge later)? Linear is simpler and probably right; branching is a power-user feature.

5. **Embedding caching strategy.** Embeddings are expensive to recompute. Cache aggressively keyed on chunk text hash + model name. Invalidate when the embedding model changes.

## Adjacent tools and prior art

Worth knowing about, none of which exactly fit this niche:

- **Prodigy** (Explosion AI). Annotation tool, mostly for NER and classification training data. Closer in spirit than most. Not free, no LLM-in-the-loop natively.
- **Label Studio.** Open source annotation. Heavyweight, web-based, more general (handles images, audio, etc.). Less focused on text chunking + rubric workflows.
- **Doccano.** Open source text annotation. Web-based. No chunking strategy choice; assumes documents are already chunked.
- **Argilla.** Modern annotation framework, integrates with LLMs. Closest to what LazyLabelText would do. Worth studying as design reference; differences would be local-first focus, chunking-as-first-class, rubric-as-artifact.
- **Kira / Evisort / Ironclad.** Commercial legal-tech with structured clause extraction. Reference for "what good labeled output looks like" but vertical-specific and proprietary.

LazyLabelText's niche: local-first, format-agnostic source documents, rubric and chunking both first-class, optimized for SME-plus-LLM iteration loops, designed to feed downstream pipelines rather than be the final destination.

## Why this is worth building

Most "label your text data" tools assume the chunking is already done and the rubric is fixed. In practice, both are iterative — you discover the right chunks by looking at the data, you discover the right rubric by trying to label and seeing where the categories fail. A tool that treats both as tunable artifacts shortens the iteration loop from days (export, edit JSON, re-import, re-run) to minutes (adjust slider, see new result).

The downstream pipelines that this enables — consolidation, RAG, training data, knowledge graphs — all depend on labeled chunks. A focused tool that does this one thing well becomes the substrate for a whole class of document-processing projects, the same way LazyLabel made image segmentation labeling tractable enough that more projects could afford to do it.

The honest version: building a labeling tool is unglamorous, the work is in interaction design and small details, and the value compounds across projects rather than landing in any one of them. That's exactly the profile of infrastructure that's underbuilt — everyone wants the downstream output, nobody wants to invest in the upstream tooling. Which is why the upstream tooling, when it exists and is good, gets used by everyone.
