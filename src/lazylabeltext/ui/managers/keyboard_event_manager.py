"""Keyboard event handling manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lazylabeltext.ui.main_window import MainWindow


class KeyboardEventManager:
    """Dispatches keyboard events based on current mode context."""

    def __init__(self, main_window: MainWindow) -> None:
        self.mw = main_window

    def handle_escape(self) -> None:
        """Cancel current operation."""
        mode = self.mw.mode_manager.current_mode
        widget = self.mw.center_panel.get_mode_widget(mode)
        if widget and hasattr(widget, "handle_escape"):
            widget.handle_escape()

    def handle_space(self) -> None:
        """Accept / confirm action (mode-dependent)."""
        mode = self.mw.mode_manager.current_mode
        if mode == "label":
            widget = self.mw.center_panel.get_mode_widget("label")
            if widget and hasattr(widget, "accept_current"):
                widget.accept_current()
