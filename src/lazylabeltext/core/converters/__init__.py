"""Document converter registry."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lazylabeltext.core.converters.base import Converter
    from lazylabeltext.core.models import ConvertedDocument

from lazylabeltext.core.exceptions import UnsupportedFormatError

logger = logging.getLogger("lazylabeltext")

CONVERTERS: dict[str, Converter] = {}

HIGH_FIDELITY_EXTENSIONS = {".pdf", ".docx"}


def _register(extensions: set[str], converter: Converter) -> None:
    """Register a converter for the given file extensions."""
    for ext in extensions:
        CONVERTERS[ext.lower()] = converter


def _docling_available() -> bool:
    return importlib.util.find_spec("docling") is not None


def convert_file(
    file_path: str,
    *,
    high_fidelity: bool = False,
    ocr_enabled: bool = False,
) -> ConvertedDocument:
    """Convert a file using the appropriate registered converter.

    When ``high_fidelity`` is True and the docling package is importable,
    PDF and DOCX files are routed through the docling-backed converter.
    Otherwise the per-extension default converter handles the file.
    """
    ext = Path(file_path).suffix.lower()

    if high_fidelity and ext in HIGH_FIDELITY_EXTENSIONS:
        if _docling_available():
            from lazylabeltext.core.converters.docling_converter import (
                DoclingConverter,
            )

            return DoclingConverter().convert(file_path, ocr_enabled=ocr_enabled)
        logger.warning(
            "high-fidelity requested but docling is not installed; "
            "falling back to default converter for %s",
            file_path,
        )

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
