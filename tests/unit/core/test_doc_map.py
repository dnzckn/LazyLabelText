"""Tests for core.doc_map."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lazylabeltext.core.doc_map import DocMap, DocMapError

SUPPORTED = {".pdf", ".docx", ".md", ".markdown", ".txt", ".text"}


def write_docmap(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "test.docmap"
    p.write_text(body, encoding="utf-8")
    return p


def make_files(root: Path, paths: list[str]) -> list[Path]:
    """Create empty files relative to ``root``. Returns absolute Paths."""
    out: list[Path] = []
    for rel in paths:
        full = root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text("")
        out.append(full)
    return out


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DocMapError):
        DocMap.load(tmp_path / "nope.docmap")


def test_parse_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    p = write_docmap(
        tmp_path,
        """
        # this is a comment
        a/*.pdf

        !drafts/**
        """,
    )
    dm = DocMap.load(p)
    assert [(e.pattern, e.is_exclude) for e in dm.entries] == [
        ("a/*.pdf", False),
        ("drafts/**", True),
    ]


def test_resolve_simple_glob(tmp_path: Path) -> None:
    make_files(tmp_path, ["a/x.pdf", "a/y.docx", "b/z.txt"])
    p = write_docmap(tmp_path, "a/*.pdf\nb/*.txt\n")
    dm = DocMap.load(p)
    res = dm.resolve(SUPPORTED)
    names = sorted(Path(f).name for f in res.files)
    assert names == ["x.pdf", "z.txt"]


def test_resolve_recursive_glob(tmp_path: Path) -> None:
    make_files(tmp_path, ["a/b/c/x.pdf", "a/y.pdf", "other.md"])
    p = write_docmap(tmp_path, "a/**/*.pdf\n")
    dm = DocMap.load(p)
    res = dm.resolve(SUPPORTED)
    assert sorted(Path(f).name for f in res.files) == ["x.pdf", "y.pdf"]


def test_exclude_overrides_include(tmp_path: Path) -> None:
    make_files(tmp_path, ["docs/keep.pdf", "docs/skip.pdf"])
    p = write_docmap(tmp_path, "docs/*.pdf\n!docs/skip.pdf\n")
    dm = DocMap.load(p)
    res = dm.resolve(SUPPORTED)
    names = sorted(Path(f).name for f in res.files)
    assert names == ["keep.pdf"]


def test_exclude_glob(tmp_path: Path) -> None:
    make_files(tmp_path, [
        "papers/published.pdf",
        "papers/draft/wip.pdf",
        "papers/draft/notes.pdf",
    ])
    p = write_docmap(tmp_path, "papers/**/*.pdf\n!papers/draft/**\n")
    dm = DocMap.load(p)
    res = dm.resolve(SUPPORTED)
    assert sorted(Path(f).name for f in res.files) == ["published.pdf"]


def test_unsupported_extensions_filtered(tmp_path: Path) -> None:
    make_files(tmp_path, ["a/keep.pdf", "a/skip.png", "a/skip.exe"])
    p = write_docmap(tmp_path, "a/*\n")
    dm = DocMap.load(p)
    res = dm.resolve(SUPPORTED)
    assert sorted(Path(f).name for f in res.files) == ["keep.pdf"]


def test_directory_target_includes_recursively(tmp_path: Path) -> None:
    make_files(tmp_path, [
        "papers/2024/a.pdf",
        "papers/2025/b.pdf",
        "papers/notes.md",
    ])
    p = write_docmap(tmp_path, "papers\n")  # a literal directory, no glob
    dm = DocMap.load(p)
    res = dm.resolve(SUPPORTED)
    assert sorted(Path(f).name for f in res.files) == ["a.pdf", "b.pdf", "notes.md"]


def test_dedupe_paths_matched_by_multiple_includes(tmp_path: Path) -> None:
    make_files(tmp_path, ["a/x.pdf"])
    p = write_docmap(tmp_path, "a/*.pdf\na/x.pdf\n")
    dm = DocMap.load(p)
    res = dm.resolve(SUPPORTED)
    assert len(res.files) == 1


def test_unmatched_pattern_reported(tmp_path: Path) -> None:
    make_files(tmp_path, ["a/x.pdf"])
    p = write_docmap(tmp_path, "a/*.pdf\nnonexistent/*.pdf\n")
    dm = DocMap.load(p)
    res = dm.resolve(SUPPORTED)
    assert len(res.files) == 1
    assert len(res.unmatched_patterns) == 1
    assert res.unmatched_patterns[0].pattern == "nonexistent/*.pdf"


def test_env_var_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_files(tmp_path, ["docs/x.pdf"])
    monkeypatch.setenv("DOCS_ROOT", str(tmp_path / "docs"))
    p = write_docmap(tmp_path, "$DOCS_ROOT/*.pdf\n")
    dm = DocMap.load(p)
    res = dm.resolve(SUPPORTED)
    assert sorted(Path(f).name for f in res.files) == ["x.pdf"]


def test_tilde_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    make_files(fake_home, ["Papers/x.pdf"])
    monkeypatch.setenv("HOME", str(fake_home))
    # Some platforms also use USERPROFILE; expanduser uses HOME on POSIX.
    p = write_docmap(tmp_path, "~/Papers/*.pdf\n")
    dm = DocMap.load(p)
    res = dm.resolve(SUPPORTED)
    if os.name == "posix":
        assert sorted(Path(f).name for f in res.files) == ["x.pdf"]


def test_absolute_path_pattern(tmp_path: Path) -> None:
    make_files(tmp_path, ["sub/x.pdf"])
    abs_pattern = str(tmp_path / "sub" / "*.pdf")
    p = write_docmap(tmp_path, abs_pattern + "\n")
    dm = DocMap.load(p)
    res = dm.resolve(SUPPORTED)
    assert sorted(Path(f).name for f in res.files) == ["x.pdf"]


def test_bang_with_no_pattern_ignored(tmp_path: Path) -> None:
    make_files(tmp_path, ["a/x.pdf"])
    p = write_docmap(tmp_path, "a/*.pdf\n!\n!  \n")
    dm = DocMap.load(p)
    res = dm.resolve(SUPPORTED)
    assert sorted(Path(f).name for f in res.files) == ["x.pdf"]
