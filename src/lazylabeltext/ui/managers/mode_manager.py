"""Mode switching manager."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lazylabeltext.ui.main_window import MainWindow

logger = logging.getLogger("lazylabeltext")

MODES = ["convert", "rubric", "chunk", "label", "results", "propagation", "export"]

# Modes where the rubric panel on the right is genuinely useful.
# Convert/Chunk/Propagation/Export don't reference categories.
# Rubric is redundant (the rubric is the center pane there).
_MODES_WITH_RUBRIC_PANEL = {"label", "results"}


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

        # Show the rubric reference panel only in modes that use it, and
        # refresh its contents whenever it becomes visible — covers the
        # case where the rubric was edited / imported since the last view.
        if hasattr(self.mw, "right_panel"):
            show_rubric = mode in _MODES_WITH_RUBRIC_PANEL
            self.mw.right_panel.setVisible(show_rubric)
            if show_rubric and getattr(self.mw, "rubric_manager", None):
                try:
                    rubric = self.mw.rubric_manager.get_active_rubric()
                    self.mw.right_panel.update_rubric(rubric)
                except Exception:
                    pass

        # Activate/deactivate mode widgets
        old_widget = self.mw.center_panel.get_mode_widget(old_mode)
        new_widget = self.mw.center_panel.get_mode_widget(mode)

        if old_widget and hasattr(old_widget, "deactivate"):
            old_widget.deactivate()
        if new_widget and hasattr(new_widget, "activate"):
            new_widget.activate()

        logger.debug("Mode: %s -> %s", old_mode, mode)
