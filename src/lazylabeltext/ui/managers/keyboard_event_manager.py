"""Keyboard event handling manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lazylabeltext.ui.main_window import MainWindow


class KeyboardEventManager:
    """Dispatches keyboard events based on current mode context.

    The labeling shortcuts (Space / C / S / F / D) need to fire both when
    the user is in label mode AND when they're in parallel mode with the
    embedded Label tab active — that's where reviewers spend most of
    their time on a labeled-corpus run, so the keys must work there too.
    """

    def __init__(self, main_window: MainWindow) -> None:
        self.mw = main_window

    # --- routing -----------------------------------------------------

    def _active_label_widget(self):
        """Return the LabelModeWidget that should receive labeling keys.

        - In label mode → the main center-panel widget.
        - In parallel mode with the Label inner tab active → the
          parallel mode's embedded label_widget.
        - Otherwise None (shortcuts are no-ops).
        """
        mode = self.mw.mode_manager.current_mode
        if mode == "label":
            return self.mw.center_panel.get_mode_widget("label")
        if mode == "parallel":
            parallel = self.mw.center_panel.get_mode_widget("parallel")
            if parallel is None:
                return None
            label_widget = getattr(parallel, "label_widget", None)
            tabs = getattr(parallel, "detail_tabs", None)
            if (
                label_widget is not None
                and tabs is not None
                and tabs.currentWidget() is label_widget
            ):
                return label_widget
        return None

    def _dispatch(self, method_name: str) -> None:
        widget = self._active_label_widget()
        if widget is None:
            return
        method = getattr(widget, method_name, None)
        if callable(method):
            method()

    # --- public handlers --------------------------------------------

    def handle_escape(self) -> None:
        """Cancel current operation (mode-dependent)."""
        mode = self.mw.mode_manager.current_mode
        widget = self.mw.center_panel.get_mode_widget(mode)
        if widget and hasattr(widget, "handle_escape"):
            widget.handle_escape()

    def handle_space(self) -> None:
        """Accept current chunk's predicted label."""
        self._dispatch("accept_current")

    def handle_correct(self) -> None:
        """Open the category picker to correct the current label."""
        self._dispatch("_correct_current")

    def handle_skip(self) -> None:
        """Skip the current chunk."""
        self._dispatch("_skip_current")

    def handle_flag(self) -> None:
        """Flag the current chunk for review."""
        self._dispatch("_flag_current")

    def handle_discard(self) -> None:
        """Discard the current chunk's label."""
        self._dispatch("_discard_current")
