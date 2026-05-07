"""Doc Map: a gitignore-syntax-but-inverted file listing source documents.

A doc map lets a project ingest documents from anywhere on disk without
copying them into a single folder. Each non-blank, non-comment line is a
path or glob pattern: includes by default, ``!``-prefixed lines exclude.

Example::

    # Research papers
    ~/Papers/2024/**/*.pdf
    ~/Notes/**/*.md
    !~/Papers/2024/draft/**

Resolution:
- ``~`` and ``$VAR`` are expanded.
- Relative patterns resolve from the doc map file's directory.
- ``**`` recurses; ``*`` matches one path segment.
- Excludes are applied after all includes are collected.
- Final list is filtered to a caller-supplied set of supported extensions.
- Duplicates (same absolute path matched by multiple includes) collapse.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("lazylabeltext")


class DocMapError(Exception):
    """Raised when a doc map file can't be read or parsed."""


@dataclass(frozen=True)
class DocMapEntry:
    """One include/exclude rule from a doc map."""

    pattern: str  # raw pattern after ``!``-prefix is stripped
    is_exclude: bool
    line_no: int  # 1-based, used in error/warning messages


@dataclass
class DocMapResolveResult:
    """Outcome of resolving a doc map against the filesystem."""

    files: list[str] = field(default_factory=list)
    # Patterns that matched zero files — surfaced to the user as warnings.
    unmatched_patterns: list[DocMapEntry] = field(default_factory=list)


class DocMap:
    """Parsed doc map ready to resolve against the filesystem."""

    def __init__(
        self,
        path: Path,
        entries: list[DocMapEntry],
    ) -> None:
        self.path = path
        self.entries = entries

    @property
    def base_dir(self) -> Path:
        """Directory used as the root for relative patterns."""
        return self.path.parent.resolve()

    @classmethod
    def load(cls, path: str | os.PathLike) -> DocMap:
        """Read and parse a doc map from disk."""
        p = Path(path)
        if not p.is_file():
            raise DocMapError(f"Doc map not found: {p}")

        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            raise DocMapError(f"Could not read doc map {p}: {e}") from e

        entries: list[DocMapEntry] = []
        for line_no, raw in enumerate(text.splitlines(), start=1):
            # Strip inline comments only when ``#`` starts the line — paths
            # can legitimately contain ``#``, and gitignore behaves the same.
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            is_exclude = line.startswith("!")
            pattern = line[1:].strip() if is_exclude else line
            if not pattern:
                # ``!`` on its own is meaningless; surface as parse warning.
                logger.warning(
                    "%s:%d empty pattern after '!' — ignored", p, line_no
                )
                continue
            entries.append(
                DocMapEntry(pattern=pattern, is_exclude=is_exclude, line_no=line_no)
            )

        return cls(path=p.resolve(), entries=entries)

    # ----------------------------------------------------------- resolution

    def _expand(self, pattern: str) -> str:
        """Apply tilde + env-var expansion to a raw pattern."""
        return os.path.expanduser(os.path.expandvars(pattern))

    def _match(self, entry: DocMapEntry) -> list[Path]:
        """Resolve one entry's pattern against the filesystem.

        Returns absolute paths. Patterns can be:
        - A literal file path (existing or not — caller decides what to do)
        - A directory (treated as ``<dir>/**/*`` so the user can write
          ``~/Papers`` and get everything inside)
        - A glob with ``*`` / ``**`` — passed to ``Path.glob`` after we
          decide whether the glob is anchored (absolute) or relative.
        """
        expanded = self._expand(entry.pattern)
        # Absolute patterns stay absolute; relative ones resolve against the
        # docmap's directory.
        if os.path.isabs(expanded):
            glob_target = expanded
        else:
            glob_target = str(self.base_dir / expanded)

        # If the (expanded, absolute) target points at an existing file or
        # directory, handle it directly — Path.glob with no wildcards
        # returns nothing useful.
        target_path = Path(glob_target)
        if "*" not in glob_target and "?" not in glob_target:
            if target_path.is_file():
                return [target_path.resolve()]
            if target_path.is_dir():
                return [
                    p.resolve() for p in target_path.rglob("*") if p.is_file()
                ]
            return []  # literal path missing

        # Glob: split into anchor + pattern. Path.glob doesn't accept
        # absolute patterns, so we identify the deepest non-glob prefix as
        # the anchor.
        parts = Path(expanded).parts if os.path.isabs(expanded) else Path(
            self.base_dir / expanded
        ).parts
        prefix_parts: list[str] = []
        for part in parts:
            if "*" in part or "?" in part:
                break
            prefix_parts.append(part)
        if not prefix_parts:
            return []
        anchor_path = Path(*prefix_parts)
        glob_rel = "/".join(parts[len(prefix_parts):])
        if not glob_rel:
            # No wildcard portion — already handled above.
            return []
        # Bare ``**`` means "everything beneath" — Path.glob treats it as
        # matching directories, so rewrite to ``**/*`` for files. Same when
        # a pattern segment ends with ``/**`` (e.g. ``papers/draft/**``).
        if glob_rel == "**" or glob_rel.endswith("/**"):
            glob_rel = glob_rel + "/*"
        try:
            return [
                p.resolve()
                for p in anchor_path.glob(glob_rel)
                if p.is_file()
            ]
        except (OSError, ValueError) as e:
            logger.warning(
                "%s:%d glob failed for %r: %s", self.path, entry.line_no,
                entry.pattern, e,
            )
            return []

    def resolve(self, supported_exts: set[str]) -> DocMapResolveResult:
        """Resolve include/exclude rules to a deduped list of absolute paths.

        ``supported_exts`` should contain extensions including the leading
        dot (``{".pdf", ".docx"}``). Files with other extensions are
        dropped — the doc map flow honors the same registry the
        folder-walk flow uses.
        """
        included: dict[Path, DocMapEntry] = {}
        excluded_patterns: list[DocMapEntry] = []
        unmatched: list[DocMapEntry] = []

        # First pass: collect includes; remember excludes for second pass.
        for entry in self.entries:
            if entry.is_exclude:
                excluded_patterns.append(entry)
                continue
            matches = self._match(entry)
            if not matches:
                unmatched.append(entry)
                continue
            for m in matches:
                if m.suffix.lower() not in supported_exts:
                    continue
                if m not in included:
                    included[m] = entry

        # Second pass: apply excludes. We resolve each exclude pattern to a
        # set of paths and remove them from `included`.
        for entry in excluded_patterns:
            for m in self._match(entry):
                included.pop(m, None)

        # Stable order: by absolute path string. Predictable for tests + UI.
        files = sorted(str(p) for p in included)
        return DocMapResolveResult(files=files, unmatched_patterns=unmatched)
