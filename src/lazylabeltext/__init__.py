"""LazyLabelText - Desktop tool for producing labeled text corpora."""

from __future__ import annotations


def _get_version() -> str:
    """Detect version through multiple fallback strategies."""
    # 1. importlib.metadata (installed package)
    try:
        from importlib.metadata import version

        return version("lazylabeltext")
    except Exception:
        pass

    # 2. Parse pyproject.toml (development mode)
    try:
        from pathlib import Path

        import tomllib

        pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
        if pyproject.exists():
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            return data["project"]["version"]
    except Exception:
        pass

    # 3. Bundled _version.py (PyInstaller)
    try:
        from ._version import __version__ as v

        return v
    except Exception:
        pass

    return "unknown"


__version__ = _get_version()
