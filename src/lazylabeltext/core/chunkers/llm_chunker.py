"""LLM-based chunker: ask the model to split text into atomic clauses."""

from __future__ import annotations

import json
import logging
import re

from lazylabeltext.core.chunkers import _register
from lazylabeltext.core.exceptions import ChunkOperationError
from lazylabeltext.core.models import Chunk, ConvertedDocument
from lazylabeltext.utils.token_counter import count_tokens

logger = logging.getLogger("lazylabeltext")

PROMPT_TEMPLATE = """Split the following text into atomic clauses. An atomic clause is a self-contained unit of meaning: one rule, one fact, one definition, one procedural step, one parameter description, one specification line, etc.

Rules:
- Return ONLY a JSON array of strings, no prose, no markdown fences.
- Each string MUST be an exact verbatim substring of the input.
- The strings must appear in the same order as in the input.
- Together they should cover the substantive content (skip pure whitespace, separator lines, navigation chrome).
- A markdown table row is one atomic clause. A bullet point is one atomic clause. A numbered step is one atomic clause. A standalone sentence is one atomic clause.
- Do NOT merge multiple atomic units into one string.
- Do NOT split a single atomic unit across multiple strings.

Input:
<<<
{text}
>>>"""


def _windows_by_tokens(text: str, max_window_tokens: int) -> list[tuple[int, int]]:
    """Split text into (char_start, char_end) windows, each ≤ max_window_tokens.

    Splits at paragraph (\\n\\n) boundaries when possible, falls back to single
    newlines, then to hard char-count splits as a last resort.
    """
    if count_tokens(text) <= max_window_tokens:
        return [(0, len(text))]

    windows: list[tuple[int, int]] = []
    cursor = 0
    paragraphs = text.split("\n\n")
    chunk_start = 0
    chunk_text_parts: list[str] = []
    chunk_tokens = 0

    for para in paragraphs:
        para_with_sep = para + "\n\n"
        para_tokens = count_tokens(para)

        if chunk_tokens + para_tokens > max_window_tokens and chunk_text_parts:
            chunk_end = chunk_start + sum(len(p) for p in chunk_text_parts)
            windows.append((chunk_start, chunk_end))
            chunk_start = chunk_end
            chunk_text_parts = []
            chunk_tokens = 0

        chunk_text_parts.append(para_with_sep)
        chunk_tokens += para_tokens
        cursor += len(para_with_sep)

    if chunk_text_parts:
        chunk_end = chunk_start + sum(len(p) for p in chunk_text_parts)
        windows.append((chunk_start, min(chunk_end, len(text))))

    # Clamp final window to text length.
    if windows and windows[-1][1] > len(text):
        s, _ = windows[-1]
        windows[-1] = (s, len(text))

    return windows or [(0, len(text))]


def _parse_clauses(response_text: str) -> list[str]:
    """Extract a JSON array of strings from an LLM response, tolerating fences."""
    text = response_text.strip()
    # Strip ```json ... ``` fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a [...] block within the response
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))

    if not isinstance(data, list):
        raise ValueError("LLM response did not contain a JSON array")
    return [s for s in data if isinstance(s, str) and s.strip()]


def _locate_clauses(window_text: str, clauses: list[str]) -> list[tuple[int, int]]:
    """Find each clause in window_text in order. Returns (start, end) char ranges."""
    positions: list[tuple[int, int]] = []
    cursor = 0
    for clause in clauses:
        # Find verbatim from cursor; fall back to whitespace-normalized match.
        idx = window_text.find(clause, cursor)
        if idx == -1:
            normalized_clause = re.sub(r"\s+", " ", clause).strip()
            normalized_window = re.sub(r"\s+", " ", window_text[cursor:])
            ni = normalized_window.find(normalized_clause)
            if ni == -1:
                logger.debug("LLM clause not found in window: %r", clause[:80])
                continue
            # Approximate: skip ahead to ni in original by counting non-whitespace.
            # (Lossy fallback — accept the approximation.)
            idx = cursor + ni
        end = idx + len(clause)
        positions.append((idx, end))
        cursor = end
    return positions


def _section_path_for(document: ConvertedDocument, char_pos: int) -> list[str]:
    for section in reversed(document.sections):
        if section.char_start <= char_pos < section.char_end:
            return section.section_path
    return []


class LLMChunker:
    """Use the LLM to identify atomic clause boundaries in a document."""

    @property
    def name(self) -> str:
        return "llm"

    def chunk(self, document: ConvertedDocument, params: dict) -> list[Chunk]:
        provider = params.get("_llm_provider")
        if provider is None:
            raise ChunkOperationError(
                "chunk",
                "llm chunker requires an LLM provider; configure one in "
                "Provider Settings.",
            )

        text = document.full_text
        if not text.strip():
            return []

        max_window_tokens = params.get("max_window_tokens", 3000)
        max_response_tokens = params.get("max_response_tokens", 4096)

        windows = _windows_by_tokens(text, max_window_tokens)
        progress_cb = params.get("_progress_callback")
        chunks: list[Chunk] = []
        doc_id = document.id or 0

        for i, (w_start, w_end) in enumerate(windows):
            window_text = text[w_start:w_end]
            if progress_cb:
                try:
                    progress_cb(i + 1, len(windows))
                except Exception:
                    pass

            prompt = PROMPT_TEMPLATE.format(text=window_text)
            try:
                response = provider.complete(prompt, max_tokens=max_response_tokens)
            except Exception as e:
                raise ChunkOperationError("chunk", f"LLM call failed: {e}") from e

            try:
                clauses = _parse_clauses(response)
            except Exception as e:
                logger.warning(
                    "Failed to parse LLM chunker response (window %d): %s", i, e
                )
                continue

            positions = _locate_clauses(window_text, clauses)
            for clause, (rel_start, rel_end) in zip(clauses, positions, strict=False):
                abs_start = w_start + rel_start
                abs_end = w_start + rel_end
                section_path = _section_path_for(document, abs_start)
                chunks.append(
                    Chunk(
                        document_id=doc_id,
                        text=clause.strip(),
                        char_start=abs_start,
                        char_end=abs_end,
                        section_path=section_path,
                        token_count=count_tokens(clause),
                        boundary_confidence=1.0,
                    )
                )

        return chunks


_register("llm", LLMChunker())
