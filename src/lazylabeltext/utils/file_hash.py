"""Content hashing for document identity.

A document's canonical identity is the SHA256 of its source bytes — not
its filename, not its path. Two ingestion paths pointing at the same
content (e.g. a folder open + a doc map opened later) collapse onto the
same row, so chunks/labels/reviews are preserved when files move.

SHA256 is chosen over faster alternatives (xxh3, BLAKE3) because:
- it's stdlib (no extra dependency),
- ~200 MB/s single-thread is invisible behind PDF parsing + LLM calls,
- users can verify against ``sha256sum`` from the shell.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# 1 MB chunks. Larger gains nothing for SHA256 + page-cached I/O; smaller
# wastes Python overhead on per-chunk update() calls.
_HASH_CHUNK_BYTES = 1 << 20


def hash_file(path: str | Path) -> str:
    """Return the lowercase hex SHA256 of a file's bytes.

    Streams the file in 1 MB chunks so multi-GB sources don't allocate the
    full content in memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(_HASH_CHUNK_BYTES)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
