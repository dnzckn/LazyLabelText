"""Notification manager: semantic wrapper around status bar."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lazylabeltext.ui.main_window import MainWindow


class NotificationManager:
    """Manages notification display through the status bar."""

    def __init__(self, main_window: MainWindow) -> None:
        self.mw = main_window

    @property
    def status_bar(self):
        return self.mw.status_bar

    def show(self, message: str, duration: int = 3000) -> None:
        self.status_bar.show_message(message, duration)

    def show_error(self, message: str, duration: int = 8000) -> None:
        self.status_bar.show_error_message(message, duration)

    def show_success(self, message: str, duration: int = 3000) -> None:
        self.status_bar.show_success_message(message, duration)

    def show_warning(self, message: str, duration: int = 5000) -> None:
        self.status_bar.show_warning_message(message, duration)
