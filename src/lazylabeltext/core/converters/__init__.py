"""Document converter registry."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lazylabeltext.core.converters.base import Converter
    from lazylabeltext.core.models import ConvertedDocument

from lazylabeltext.core.exceptions import UnsupportedFormatError

CONVERTERS: dict[str, Converter] = {}


def _register(extensions: set[str], converter: Converter) -> None:
    """Register a converter for the given file extensions."""
    for ext in extensions:
        CONVERTERS[ext.lower()] = converter


def convert_file(file_path: str) -> ConvertedDocument:
    """Convert a file using the appropriate registered converter."""
    ext = Path(file_path).suffix.lower()
    converter = CONVERTERS.get(ext)
    if converter is None:
        raise UnsupportedFormatError(file_path, ext)
    return converter.convert(file_path)


def supported_extensions() -> set[str]:
    """Return all supported file extensions."""
    return set(CONVERTERS.keys())


# Import submodules to trigger registration
from lazylabeltext.core.converters import (  # noqa: E402, F401
    docx_converter,
    markdown_converter,
    pdf_converter,
    text_converter,
)
