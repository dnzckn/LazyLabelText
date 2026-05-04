"""Base converter protocol and shared utilities."""

from __future__ import annotations

from typing import Protocol

from lazylabeltext.core.models import ConvertedDocument


class Converter(Protocol):
    """Protocol for document format converters."""

    def convert(self, file_path: str) -> ConvertedDocument: ...

    def supported_extensions(self) -> set[str]: ...
