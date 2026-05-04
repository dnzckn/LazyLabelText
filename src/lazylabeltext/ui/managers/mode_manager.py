"""Mode switching manager."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lazylabeltext.ui.main_window import MainWindow

logger = logging.getLogger("lazylabeltext")

MODES = ["convert", "rubric", "chunk", "label", "results", "propagation", "export"]


class ModeManager:
    """Manages UI mode switching."""

    def __init__(self, main_window: MainWindow) -> None:
        self.mw = main_window
        self.current_mode = "convert"

    def set_mode(self, mode: str) -> None:
        """Switch to the specified mode."""
        if mode not in MODES:
            return

        old_mode = self.current_mode
        self.current_mode = mode

        # Update center panel
        self.mw.center_panel.set_mode(mode)

        # Update toolbar button states
        for btn_mode, btn in self.mw._mode_buttons.items():
            btn.setChecked(btn_mode == mode)

        # Activate/deactivate mode widgets
        old_widget = self.mw.center_panel.get_mode_widget(old_mode)
        new_widget = self.mw.center_panel.get_mode_widget(mode)

        if old_widget and hasattr(old_widget, "deactivate"):
            old_widget.deactivate()
        if new_widget and hasattr(new_widget, "activate"):
            new_widget.activate()

        logger.debug("Mode: %s -> %s", old_mode, mode)
