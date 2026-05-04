"""Base mode widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from lazylabeltext.core.app_context import AppContext


class BaseMode(QWidget):
    """Base class for mode-specific views."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = context

    def activate(self) -> None:
        """Called when this mode becomes active."""

    def deactivate(self) -> None:
        """Called when switching away from this mode."""

    def on_document_selected(self, doc_id: int) -> None:
        """Called when user selects a document in the left panel."""

    def refresh(self) -> None:
        """Refresh the mode's display."""
