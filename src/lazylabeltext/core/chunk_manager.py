"""Chunk management: orchestrates chunking, merge, and split."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from lazylabeltext.core.chunkers import available_strategies, chunk_document
from lazylabeltext.core.database import Database
from lazylabeltext.core.exceptions import ChunkNotFoundError, ChunkOperationError
from lazylabeltext.core.models import Chunk, ChunkingRun
from lazylabeltext.utils.token_counter import count_tokens

logger = logging.getLogger("lazylabeltext")


class ChunkManager:
    """Orchestrates chunking operations."""

    def __init__(
        self, database: Database, embedding_provider=None, llm_provider=None
    ) -> None:
        self.db = database
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider

    def set_embedding_provider(self, provider) -> None:
        self.embedding_provider = provider

    def set_llm_provider(self, provider) -> None:
        self.llm_provider = provider

    def run_chunking(
        self, document_id: int, strategy: str, params: dict
    ) -> ChunkingRun:
        """Run a chunking strategy on a document. Replaces any prior chunks."""
        doc = self.db.get_document(document_id)
        if doc is None:
            raise ChunkOperationError("chunk", f"Document {document_id} not found")

        # Re-chunking replaces prior runs/chunks/labels/reviews for this doc.
        self.db.delete_all_chunking_for_document(document_id)

        # Persisted params: drop runtime-injected internals (callbacks, providers).
        persisted_params = {k: v for k, v in params.items() if not k.startswith("_")}

        # Create the chunking run record
        run = ChunkingRun(
            document_id=document_id,
            strategy=strategy,
            params=persisted_params,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        run.id = self.db.insert_chunking_run(run)

        # Inject providers for chunkers that need them.
        chunker_params = dict(params)
        chunker_params["_embedding_provider"] = self.embedding_provider
        chunker_params["_llm_provider"] = self.llm_provider

        # Run the chunker
        chunks = chunk_document(doc, strategy, chunker_params)

        # Store chunks with run reference
        for chunk in chunks:
            chunk.document_id = document_id
            chunk.chunking_run_id = run.id  # type: ignore[assignment]
            chunk.id = self.db.insert_chunk(chunk)

        # Update run count
        run.n_chunks = len(chunks)
        self.db.update_chunking_run_count(run.id, len(chunks))  # type: ignore[arg-type]

        logger.info(
            "Chunked document %d with '%s': %d chunks",
            document_id,
            strategy,
            len(chunks),
        )
        return run

    def get_chunks(self, document_id: int, run_id: int | None = None) -> list[Chunk]:
        if run_id is not None:
            return self.db.get_chunks_for_run(run_id)
        return self.db.get_chunks_for_document(document_id)

    def merge_chunks(self, chunk_id_a: int, chunk_id_b: int) -> Chunk:
        """Merge two adjacent chunks into one."""
        a = self.db.get_chunk(chunk_id_a)
        b = self.db.get_chunk(chunk_id_b)
        if a is None:
            raise ChunkNotFoundError(chunk_id_a)
        if b is None:
            raise ChunkNotFoundError(chunk_id_b)
        if a.document_id != b.document_id:
            raise ChunkOperationError("merge", "Chunks from different documents")

        # Ensure a comes before b
        if a.char_start > b.char_start:
            a, b = b, a

        merged_text = a.text + "\n\n" + b.text
        a.text = merged_text
        a.char_end = b.char_end
        a.token_count = count_tokens(merged_text)
        a.manual_override = {"action": "merge", "merged_chunk_id": b.id}

        self.db.update_chunk(a)
        # Delete chunk b
        self.db.conn.execute("DELETE FROM chunks WHERE id = ?", (b.id,))
        self.db.conn.commit()

        logger.info("Merged chunks %d + %d", chunk_id_a, chunk_id_b)
        return a

    def split_chunk(self, chunk_id: int, char_offset: int) -> tuple[Chunk, Chunk]:
        """Split a chunk at the given character offset within the chunk text."""
        chunk = self.db.get_chunk(chunk_id)
        if chunk is None:
            raise ChunkNotFoundError(chunk_id)
        if char_offset <= 0 or char_offset >= len(chunk.text):
            raise ChunkOperationError("split", "Invalid split offset")

        text_a = chunk.text[:char_offset].strip()
        text_b = chunk.text[char_offset:].strip()

        if not text_a or not text_b:
            raise ChunkOperationError("split", "Split would create empty chunk")

        # Update original chunk to be the first half
        chunk.text = text_a
        chunk.char_end = chunk.char_start + len(text_a)
        chunk.token_count = count_tokens(text_a)
        chunk.manual_override = {"action": "split", "offset": char_offset}
        self.db.update_chunk(chunk)

        # Create new chunk for the second half
        new_chunk = Chunk(
            document_id=chunk.document_id,
            chunking_run_id=chunk.chunking_run_id,
            text=text_b,
            char_start=chunk.char_end,
            char_end=chunk.char_end + len(text_b),
            section_path=chunk.section_path,
            token_count=count_tokens(text_b),
            manual_override={"action": "split_result", "original_chunk_id": chunk.id},
        )
        new_chunk.id = self.db.insert_chunk(new_chunk)

        logger.info("Split chunk %d at offset %d", chunk_id, char_offset)
        return chunk, new_chunk

    def get_chunking_runs(self, document_id: int) -> list[ChunkingRun]:
        return self.db.get_chunking_runs(document_id)

    def delete_run(self, run_id: int) -> None:
        """Delete a chunking run and its chunks."""
        self.db.delete_chunks_for_run(run_id)
        self.db.conn.execute("DELETE FROM chunking_runs WHERE id = ?", (run_id,))
        self.db.conn.commit()

    def get_chunk_statistics(self, run_id: int) -> dict:
        """Get statistics for a chunking run."""
        chunks = self.db.get_chunks_for_run(run_id)
        if not chunks:
            return {"count": 0, "token_counts": [], "mean": 0, "median": 0}

        token_counts = [c.token_count for c in chunks]
        token_counts.sort()
        mean = sum(token_counts) / len(token_counts)
        mid = len(token_counts) // 2
        median = (
            token_counts[mid]
            if len(token_counts) % 2
            else (token_counts[mid - 1] + token_counts[mid]) / 2
        )

        return {
            "count": len(chunks),
            "token_counts": token_counts,
            "mean": round(mean, 1),
            "median": median,
            "min": min(token_counts),
            "max": max(token_counts),
        }

    @staticmethod
    def available_strategies() -> list[str]:
        return available_strategies()
