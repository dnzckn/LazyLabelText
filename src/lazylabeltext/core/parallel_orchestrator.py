"""Parallel-mode orchestrator: stage-based, per-doc state, optional concurrency.

Replaces the old PropagationManager. Operates the same three stages
(convert / chunk / label) but exposes them independently so the user can
review the corpus state between stages, drop docs out of an in-flight run,
and re-run a stage on a single doc with custom params without disturbing
the rest of the corpus.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from lazylabeltext.config.settings import Settings
    from lazylabeltext.core.chunk_manager import ChunkManager
    from lazylabeltext.core.database import Database
    from lazylabeltext.core.document_manager import DocumentManager
    from lazylabeltext.core.label_manager import LabelManager
    from lazylabeltext.core.models import Rubric

logger = logging.getLogger("lazylabeltext")

Stage = Literal["convert", "chunk", "label"]
DocStage = Literal[
    "idle", "convert", "chunk", "label", "queued",
    "done", "failed", "cancelled", "excluded",
]


@dataclass
class DocState:
    """Per-document state visible to the UI strip."""

    doc_id: int
    filename: str
    included: bool = True
    stage: DocStage = "idle"
    progress: float = 0.0  # 0..1 within current stage
    message: str = ""
    error: str | None = None
    chunks: int = 0
    labels: int = 0


@dataclass
class StageResult:
    stage: Stage
    total: int
    succeeded: int
    failed: int
    cancelled: bool
    artifacts: int  # docs/chunks/labels created or replaced
    errors: list[dict] = field(default_factory=list)


@dataclass
class Callbacks:
    """Hooks used by the orchestrator to push updates back to the UI worker."""

    on_doc_state: Callable[[DocState], None] | None = None
    on_stage_progress: Callable[[int, int, str], None] | None = None


class ParallelOrchestrator:
    def __init__(
        self,
        database: Database,
        document_manager: DocumentManager,
        chunk_manager: ChunkManager,
        label_manager: LabelManager,
        settings: Settings | None = None,
    ) -> None:
        self.db = database
        self.document_manager = document_manager
        self.chunk_manager = chunk_manager
        self.label_manager = label_manager
        self.settings = settings

        self._states: dict[int, DocState] = {}
        # RLock not Lock: the label-stage finalize updates per-doc state
        # under this lock and then calls _emit (which also acquires the
        # lock to snapshot the state). With a non-reentrant Lock that
        # second acquire deadlocks the worker thread, freezing the GUI
        # next time it tries set_inclusion (e.g. when the user clicks
        # Cancel). RLock makes the re-entry from the same thread safe.
        self._states_lock = threading.RLock()
        self._cancel_all = threading.Event()
        # Track which docs are actively in a worker right now so cancel_all
        # can immediately mark *queued* (not-yet-started) docs as cancelled
        # without waiting for in-flight LLM calls to finish.
        self._running_doc_ids: set[int] = set()
        self._running_lock = threading.Lock()
        # Callbacks for the currently-running stage; used by cancel_all to
        # emit state updates for the docs it transitions to "cancelled".
        self._active_callbacks: Callbacks | None = None

    # ------------------------------------------------------------------ state

    def refresh_states(self) -> list[DocState]:
        """Rebuild per-doc state from the database (called when entering mode)."""
        docs = self.document_manager.get_all_documents()
        status = {s["filename"]: s for s in self.db.get_document_label_status()}
        with self._states_lock:
            for d in docs:
                if d.id is None:
                    continue
                existing = self._states.get(d.id)
                included = existing.included if existing else True
                s = status.get(d.filename, {})
                state = DocState(
                    doc_id=d.id,
                    filename=d.filename,
                    included=included,
                    stage=existing.stage if existing else "idle",
                    chunks=int(s.get("chunk_count", 0)),
                    labels=int(s.get("label_count", 0)),
                )
                if d.status == "failed":
                    state.stage = "failed"
                    state.error = d.warnings[0] if d.warnings else "convert failed"
                self._states[d.id] = state
            # Drop states for docs no longer in the project
            live_ids = {d.id for d in docs if d.id is not None}
            for stale in list(self._states):
                if stale not in live_ids:
                    self._states.pop(stale)
            return list(self._states.values())

    def get_states(self) -> list[DocState]:
        with self._states_lock:
            return list(self._states.values())

    def set_inclusion(self, doc_id: int, included: bool) -> DocState | None:
        with self._states_lock:
            s = self._states.get(doc_id)
            if s is None:
                return None
            s.included = included
            if not included and s.stage in ("idle", "done"):
                # Visual hint only — actual skipping is by reading included
                # flag at run_stage time; don't overwrite a meaningful stage.
                pass
            return s

    # --------------------------------------------------------------- helpers

    def _emit(self, callbacks: Callbacks | None, doc_id: int) -> None:
        if callbacks is None or callbacks.on_doc_state is None:
            return
        with self._states_lock:
            s = self._states.get(doc_id)
            if s is None:
                return
            snapshot = DocState(**vars(s))
        callbacks.on_doc_state(snapshot)

    def _set_stage(
        self,
        doc_id: int,
        stage: DocStage,
        *,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._states_lock:
            s = self._states.get(doc_id)
            if s is None:
                return
            s.stage = stage
            if progress is not None:
                s.progress = progress
            if message is not None:
                s.message = message
            if error is not None:
                s.error = error

    def _per_doc_cancelled(self, doc_id: int) -> bool:
        # A doc is considered cancelled if it's been excluded mid-run; we
        # repurpose the included flag (set by the UI strip's "Drop") and the
        # global cancel flag.
        if self._cancel_all.is_set():
            return True
        with self._states_lock:
            s = self._states.get(doc_id)
            if s is None:
                return True
            return not s.included

    def cancel_all(self) -> None:
        """Request cancel + immediately mark queued docs as cancelled.

        Running docs (those a worker has already entered) keep their
        current state until the worker hits the next chunk-boundary check
        and bails — Python can't interrupt the in-flight LLM call. But
        queued docs (still waiting in the executor's pool) get visibly
        cancelled in the strip *now* instead of after the executor drains.
        """
        self._cancel_all.set()
        with self._running_lock:
            running = set(self._running_doc_ids)
        cancelled_ids: list[int] = []
        with self._states_lock:
            for doc_id, s in self._states.items():
                if doc_id in running:
                    continue
                if s.stage in ("queued", "label", "chunk", "convert"):
                    s.stage = "cancelled"
                    s.message = "cancelled before run"
                    s.progress = 0.0
                    cancelled_ids.append(doc_id)
        cb = self._active_callbacks
        if cb is not None and cb.on_doc_state is not None:
            for doc_id in cancelled_ids:
                self._emit(cb, doc_id)

    def cancel_doc(self, doc_id: int) -> None:
        # Same flag as exclude — orchestrator inspects this at the next
        # safe point (between docs / between chunks).
        self.set_inclusion(doc_id, included=False)

    def _reset_cancel(self) -> None:
        self._cancel_all.clear()

    # ------------------------------------------------------------- run_stage

    def run_stage(
        self,
        stage: Stage,
        doc_ids: list[int],
        params: dict,
        max_workers: int,
        callbacks: Callbacks | None = None,
    ) -> StageResult:
        self._reset_cancel()
        with self._running_lock:
            self._running_doc_ids.clear()
        self._active_callbacks = callbacks
        try:
            # Filter: only included, alive states. Failed-convert docs can
            # still have convert re-run on them; excluded docs are skipped.
            with self._states_lock:
                targets = [
                    self._states[d].doc_id
                    for d in doc_ids
                    if d in self._states and self._states[d].included
                ]

            if stage == "convert":
                return self._run_convert(targets, params, max_workers, callbacks)
            if stage == "chunk":
                return self._run_chunk(targets, params, max_workers, callbacks)
            if stage == "label":
                return self._run_label(targets, params, max_workers, callbacks)
            raise ValueError(f"Unknown stage: {stage}")
        finally:
            self._active_callbacks = None
            with self._running_lock:
                self._running_doc_ids.clear()

    def run_all_stages(
        self,
        doc_ids: list[int],
        params: dict,
        max_workers: int,
        callbacks: Callbacks | None = None,
    ) -> list[StageResult]:
        results: list[StageResult] = []
        for stage in ("convert", "chunk", "label"):
            r = self.run_stage(stage, doc_ids, params, max_workers, callbacks)  # type: ignore[arg-type]
            results.append(r)
            if r.cancelled:
                break
        return results

    # ----------------------------------------------------------- convert all

    def _run_convert(
        self,
        doc_ids: list[int],
        params: dict,
        max_workers: int,
        callbacks: Callbacks | None,
    ) -> StageResult:
        result = StageResult(
            stage="convert", total=len(doc_ids), succeeded=0, failed=0,
            cancelled=False, artifacts=0,
        )
        if not doc_ids:
            return result

        max_workers = max(1, int(max_workers))

        def worker(doc_id: int) -> tuple[int, bool, str | None]:
            if self._per_doc_cancelled(doc_id):
                self._set_stage(doc_id, "cancelled", message="dropped from run")
                self._emit(callbacks, doc_id)
                return doc_id, False, "cancelled"
            self._set_stage(doc_id, "convert", progress=0.0, message="converting…")
            self._emit(callbacks, doc_id)
            try:
                self.document_manager.reconvert(doc_id)
            except Exception as e:
                self._set_stage(doc_id, "failed", error=str(e), message="convert failed")
                self._emit(callbacks, doc_id)
                return doc_id, False, str(e)
            self._set_stage(doc_id, "done", progress=1.0, message="converted")
            self._emit(callbacks, doc_id)
            return doc_id, True, None

        return self._run_executor(worker, doc_ids, max_workers, result, callbacks)

    # ------------------------------------------------------------- chunk all

    def _run_chunk(
        self,
        doc_ids: list[int],
        params: dict,
        max_workers: int,
        callbacks: Callbacks | None,
    ) -> StageResult:
        strategy = params.get("strategy", "structural")
        chunk_params = params.get("chunk_params", {}) or {}
        result = StageResult(
            stage="chunk", total=len(doc_ids), succeeded=0, failed=0,
            cancelled=False, artifacts=0,
        )
        if not doc_ids:
            return result

        max_workers = max(1, int(max_workers))
        artifacts_lock = threading.Lock()

        def worker(doc_id: int) -> tuple[int, bool, str | None]:
            if self._per_doc_cancelled(doc_id):
                self._set_stage(doc_id, "cancelled", message="dropped from run")
                self._emit(callbacks, doc_id)
                return doc_id, False, "cancelled"
            self._set_stage(doc_id, "chunk", progress=0.0, message="chunking…")
            self._emit(callbacks, doc_id)
            try:
                run = self.chunk_manager.run_chunking(doc_id, strategy, chunk_params)
                chunks = self.chunk_manager.get_chunks(doc_id, run.id)
                with self._states_lock:
                    s = self._states.get(doc_id)
                    if s is not None:
                        s.chunks = len(chunks)
                with artifacts_lock:
                    result.artifacts += len(chunks)
            except Exception as e:
                self._set_stage(doc_id, "failed", error=str(e), message="chunk failed")
                self._emit(callbacks, doc_id)
                return doc_id, False, str(e)
            self._set_stage(
                doc_id, "done", progress=1.0,
                message=f"{len(chunks)} chunks",
            )
            self._emit(callbacks, doc_id)
            return doc_id, True, None

        return self._run_executor(worker, doc_ids, max_workers, result, callbacks)

    # ------------------------------------------------------------- label all

    def _run_label(
        self,
        doc_ids: list[int],
        params: dict,
        max_workers: int,
        callbacks: Callbacks | None,
    ) -> StageResult:
        rubric: Rubric | None = params.get("rubric")
        result = StageResult(
            stage="label", total=len(doc_ids), succeeded=0, failed=0,
            cancelled=False, artifacts=0,
        )
        if rubric is None or not doc_ids:
            return result
        if self.label_manager.llm_provider is None:
            for doc_id in doc_ids:
                self._set_stage(
                    doc_id, "failed",
                    error="LLM provider not configured",
                    message="no LLM",
                )
                self._emit(callbacks, doc_id)
            result.failed = len(doc_ids)
            result.errors = [
                {"doc_id": d, "error": "LLM provider not configured"} for d in doc_ids
            ]
            return result

        max_workers = max(1, int(max_workers))
        artifacts_lock = threading.Lock()

        # Per-doc parallelism: each worker takes one whole doc and labels
        # its chunks sequentially. With workers=2, two docs label in
        # parallel; whichever finishes first picks up the next doc. The
        # reviewer can start reviewing the first finished doc while the
        # rest are still in flight — better for review-driven workflows
        # than spreading workers across all docs (which makes everything
        # finish around the same time).
        # Use only the *latest* chunking run's chunks per doc — calling
        # get_chunks(doc_id) without a run_id returns every chunk ever
        # created, including stale ones, which would disagree with
        # chunk-mode (which always shows the latest run).
        # Resume support: chunks already labeled for this rubric should be
        # skipped on a re-run, not relabeled. Build a set of labeled chunk
        # ids once, up front. After cancel + restart, the worker walks the
        # chunk list as before but skips any chunk in this set, so the run
        # picks up where the previous one left off.
        labeled_chunk_ids: set[int] = set()
        if rubric.id is not None:
            for label in self.db.get_all_labels(rubric.id):
                if label.chunk_id is not None:
                    labeled_chunk_ids.add(label.chunk_id)

        chunks_by_doc: dict[int, list] = {}
        per_doc_total: dict[int, int] = {}
        per_doc_resumed: dict[int, int] = {}
        for doc_id in doc_ids:
            runs = self.chunk_manager.get_chunking_runs(doc_id)
            if runs:
                latest_run = runs[-1]
                chunks = self.chunk_manager.get_chunks(doc_id, latest_run.id)
            else:
                chunks = []
            chunks_by_doc[doc_id] = chunks
            per_doc_total[doc_id] = len(chunks)
            already = sum(
                1 for ch in chunks if ch.id is not None and ch.id in labeled_chunk_ids
            )
            per_doc_resumed[doc_id] = already
            if not chunks:
                self._set_stage(
                    doc_id, "done", progress=1.0, message="no chunks to label",
                )
                self._emit(callbacks, doc_id)
            elif already == len(chunks):
                # All chunks already labeled — nothing to do for this doc.
                self._set_stage(
                    doc_id, "done", progress=1.0,
                    message=f"{already} labeled",
                )
                self._emit(callbacks, doc_id)
            else:
                # "queued" instead of "label" so the strip badge can
                # distinguish docs waiting in the executor pool from ones
                # a worker is actively labeling. doc_worker flips the
                # stage to "label" when it starts processing this doc.
                progress0 = already / len(chunks) if chunks else 0.0
                msg = (
                    f"queued {len(chunks) - already} of {len(chunks)}"
                    + (f" ({already} already labeled)" if already else "")
                )
                self._set_stage(
                    doc_id, "queued", progress=progress0, message=msg,
                )
                self._emit(callbacks, doc_id)

        # Only docs that have at least one *unlabeled* chunk get a worker.
        runnable_doc_ids = [
            d for d in doc_ids
            if chunks_by_doc.get(d)
            and per_doc_resumed.get(d, 0) < per_doc_total.get(d, 0)
        ]
        if not runnable_doc_ids:
            return result

        # per_doc_done counts total labeled chunks (resumed + this run) so
        # the progress bar reflects the doc's *overall* labeling state, not
        # just this run's contribution. Initialise with the resumed count.
        per_doc_done: dict[int, int] = dict(per_doc_resumed)
        per_doc_failed: dict[int, int] = dict.fromkeys(per_doc_total, 0)

        def doc_worker(doc_id: int) -> tuple[int, bool, str | None]:
            """Label every chunk of one doc sequentially, in order."""
            if self._per_doc_cancelled(doc_id):
                with self._states_lock:
                    s = self._states.get(doc_id)
                    if s and s.stage != "cancelled":
                        s.stage = "cancelled"
                        s.message = "dropped from run"
                self._emit(callbacks, doc_id)
                return doc_id, False, "cancelled"

            # Mark this doc "running" so cancel_all knows to leave its state
            # alone (the worker will set the final state when it bails out
            # at the next chunk boundary).
            with self._running_lock:
                self._running_doc_ids.add(doc_id)
            try:
                return _label_doc_body(doc_id)
            finally:
                with self._running_lock:
                    self._running_doc_ids.discard(doc_id)

        def _label_doc_body(doc_id: int) -> tuple[int, bool, str | None]:
            chunks = chunks_by_doc.get(doc_id, [])
            total_for_doc = per_doc_total[doc_id]
            # Flip queued → label now that we're actually processing this
            # doc. The strip badge changes from "queued" to "labeling" so
            # the user can see which docs are in flight vs waiting.
            done0 = per_doc_done[doc_id]
            self._set_stage(
                doc_id, "label",
                progress=(done0 / total_for_doc) if total_for_doc else 0.0,
                message=f"labeling {done0}/{total_for_doc}",
            )
            self._emit(callbacks, doc_id)
            last_err: str | None = None
            for ch in chunks:
                # Skip chunks that already have a label for this rubric —
                # this is what makes cancel + resume pick up where it left
                # off instead of relabeling everything from scratch.
                if ch.id is not None and ch.id in labeled_chunk_ids:
                    continue
                # Honor cancel between chunks so the user can pull a doc
                # out of an in-flight run without waiting for it to finish.
                if self._per_doc_cancelled(doc_id):
                    # Update state explicitly so the strip never sees a
                    # bailed-mid-doc row stuck on its previous "labeling
                    # K/N" badge until finalize runs (which is also after
                    # all *other* workers finish).
                    done = per_doc_done[doc_id]
                    fail = per_doc_failed[doc_id]
                    self._set_stage(
                        doc_id, "cancelled",
                        progress=(done + fail) / total_for_doc if total_for_doc else 0.0,
                        message=(
                            f"cancelled at {done + fail}/{total_for_doc}"
                            + (f", {done} labeled" if done else "")
                        ),
                    )
                    self._emit(callbacks, doc_id)
                    return doc_id, False, "cancelled"
                try:
                    self.label_manager.label_chunk(ch, rubric)
                    per_doc_done[doc_id] += 1
                    with artifacts_lock:
                        result.artifacts += 1
                except Exception as e:
                    per_doc_failed[doc_id] += 1
                    last_err = str(e)
                done = per_doc_done[doc_id]
                fail = per_doc_failed[doc_id]
                self._set_stage(
                    doc_id, "label",
                    progress=(done + fail) / total_for_doc if total_for_doc else 1.0,
                    message=(
                        f"labeling {done + fail}/{total_for_doc}"
                        + (f" ({fail} err)" if fail else "")
                    ),
                )
                self._emit(callbacks, doc_id)
            if per_doc_failed[doc_id] and per_doc_done[doc_id] == 0:
                return doc_id, False, last_err or "all chunks failed"
            return doc_id, True, None

        total_docs = len(runnable_doc_ids)
        completed_docs = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(doc_worker, d): d for d in runnable_doc_ids}
            try:
                for fut in as_completed(futs):
                    if self._cancel_all.is_set():
                        result.cancelled = True
                        for f in futs:
                            f.cancel()
                        break
                    doc_id, ok, err = fut.result()
                    completed_docs += 1
                    if callbacks and callbacks.on_stage_progress:
                        callbacks.on_stage_progress(
                            completed_docs, total_docs,
                            f"label {completed_docs}/{total_docs} docs",
                        )
                    if not ok and err and err != "cancelled":
                        result.errors.append({"doc_id": doc_id, "error": err})
            finally:
                if result.cancelled:
                    ex.shutdown(wait=False, cancel_futures=True)

        # Finalize per-doc state. If the run was cancelled and a doc didn't
        # get all its chunks labeled, mark it "cancelled" with a partial
        # count — not "done", which would falsely imply completion.
        for doc_id, total_for_doc in per_doc_total.items():
            done = per_doc_done.get(doc_id, 0)
            fail = per_doc_failed.get(doc_id, 0)
            attempted = done + fail
            with self._states_lock:
                s = self._states.get(doc_id)
                if s is None:
                    continue
                if s.stage == "cancelled":
                    # Per-doc drop already set this; preserve and just
                    # update the count for the strip.
                    s.labels = done
                    self._emit(callbacks, doc_id)
                    continue
                s.labels = done
                if result.cancelled and attempted < total_for_doc:
                    s.stage = "cancelled"
                    s.message = (
                        f"cancelled at {attempted}/{total_for_doc}"
                        + (f", {done} labeled" if done else "")
                    )
                    s.progress = (
                        attempted / total_for_doc if total_for_doc else 0.0
                    )
                elif fail and done == 0:
                    s.stage = "failed"
                    s.message = f"{fail} labels failed"
                else:
                    s.stage = "done"
                    s.message = f"{done} labeled" + (
                        f", {fail} failed" if fail else ""
                    )
                    s.progress = 1.0
            self._emit(callbacks, doc_id)
        result.succeeded = sum(per_doc_done.values())
        result.failed = sum(per_doc_failed.values())
        return result

    # ----------------------------------------------------- generic executor

    def _run_executor(
        self,
        worker: Callable[[int], tuple[int, bool, str | None]],
        doc_ids: list[int],
        max_workers: int,
        result: StageResult,
        callbacks: Callbacks | None,
    ) -> StageResult:
        completed = 0
        total = len(doc_ids)
        completed_doc_ids: set[int] = set()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(worker, d): d for d in doc_ids}
            try:
                for fut in as_completed(futs):
                    if self._cancel_all.is_set():
                        result.cancelled = True
                        break
                    doc_id, ok, err = fut.result()
                    completed += 1
                    completed_doc_ids.add(doc_id)
                    if ok:
                        result.succeeded += 1
                        result.artifacts = max(result.artifacts, result.succeeded)
                    else:
                        if err == "cancelled":
                            pass
                        else:
                            result.failed += 1
                            result.errors.append({"doc_id": doc_id, "error": err})
                    if callbacks and callbacks.on_stage_progress:
                        callbacks.on_stage_progress(
                            completed, total,
                            f"{result.stage} {completed}/{total}",
                        )
            finally:
                if result.cancelled:
                    ex.shutdown(wait=False, cancel_futures=True)

        # If cancelled, mark any doc that didn't actually finish back to
        # "cancelled" so the strip doesn't show stale "convert…"/"chunk…"
        # transitional states forever.
        if result.cancelled:
            for d in doc_ids:
                if d in completed_doc_ids:
                    continue
                with self._states_lock:
                    s = self._states.get(d)
                    if s is None:
                        continue
                    if s.stage in (result.stage, "idle"):
                        s.stage = "cancelled"
                        s.message = "cancelled before run"
                        s.progress = 0.0
                self._emit(callbacks, d)
        return result

    # ----------------------------------------------------------- rerun_one

    def rerun_one(
        self,
        doc_id: int,
        stage: Stage,
        params: dict,
        callbacks: Callbacks | None = None,
    ) -> StageResult:
        """Re-run a single stage for one doc with custom params.

        Used by the drill-in panel when the user wants to fine-tune one doc
        without disturbing the rest of the corpus. Sequential, no executor.
        Inclusion flag is honored — re-run on excluded doc is a no-op.
        """
        self._reset_cancel()
        with self._states_lock:
            s = self._states.get(doc_id)
            if s is None or not s.included:
                return StageResult(
                    stage=stage, total=0, succeeded=0, failed=0,
                    cancelled=False, artifacts=0,
                )
        return self.run_stage(stage, [doc_id], params, max_workers=1, callbacks=callbacks)
