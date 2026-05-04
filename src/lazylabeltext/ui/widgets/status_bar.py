"""Custom status bar with theme toggle and semantic messages."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QStatusBar, QWidget


class StatusBar(QStatusBar):
    """Custom status bar with themed messages and project info."""

    theme_toggled = pyqtSignal(bool)

    _COLORS = {
        "message": ("#ffa500", "#c47600"),
        "error": ("#ff6b6b", "#c62828"),
        "success": ("#51cf66", "#2e7d32"),
        "warning": ("#ffd43b", "#b8860b"),
    }

    def __init__(self, dark_mode: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dark_mode = dark_mode
        self._message_timer = QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.timeout.connect(self._clear_message)
        self._setup_ui()

    def _setup_ui(self) -> None:
        # Theme toggle
        self.theme_toggle = QCheckBox("Dark")
        self.theme_toggle.setChecked(self._dark_mode)
        self.theme_toggle.toggled.connect(self.theme_toggled.emit)
        theme_container = QWidget()
        theme_layout = QHBoxLayout(theme_container)
        theme_layout.setContentsMargins(4, 0, 8, 0)
        theme_layout.addWidget(self.theme_toggle)
        self.addWidget(theme_container)

        # Main message label
        self.message_label = QLabel()
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.addWidget(self.message_label, 1)

        # Project stats (permanent right side)
        self.stats_label = QLabel()
        self.addPermanentWidget(self.stats_label)

        # Provider indicator
        self.provider_label = QLabel()
        self.addPermanentWidget(self.provider_label)

    def _color(self, key: str) -> str:
        dark, light = self._COLORS[key]
        return dark if self._dark_mode else light

    def show_message(self, message: str, duration: int = 3000) -> None:
        self.message_label.setText(message)
        self.message_label.setStyleSheet(
            f"color: {self._color('message')}; padding: 2px 5px;"
        )
        self._message_timer.stop()
        if duration > 0:
            self._message_timer.start(duration)

    def show_error_message(self, message: str, duration: int = 8000) -> None:
        self.message_label.setText(message)
        self.message_label.setStyleSheet(
            f"color: {self._color('error')}; font-weight: bold; padding: 2px 5px;"
        )
        self._message_timer.stop()
        if duration > 0:
            self._message_timer.start(duration)

    def show_success_message(self, message: str, duration: int = 3000) -> None:
        self.message_label.setText(message)
        self.message_label.setStyleSheet(
            f"color: {self._color('success')}; padding: 2px 5px;"
        )
        self._message_timer.stop()
        if duration > 0:
            self._message_timer.start(duration)

    def show_warning_message(self, message: str, duration: int = 5000) -> None:
        self.message_label.setText(message)
        self.message_label.setStyleSheet(
            f"color: {self._color('warning')}; padding: 2px 5px;"
        )
        self._message_timer.stop()
        if duration > 0:
            self._message_timer.start(duration)

    def set_project_stats(
        self, docs: int = 0, chunks: int = 0, labels: int = 0
    ) -> None:
        self.stats_label.setText(
            f"Docs: {docs}  |  Chunks: {chunks}  |  Labels: {labels}"
        )

    def set_provider_status(self, text: str) -> None:
        self.provider_label.setText(text)

    def update_theme(self, dark_mode: bool) -> None:
        self._dark_mode = dark_mode
        self.theme_toggle.setChecked(dark_mode)

    def _clear_message(self) -> None:
        self.message_label.clear()
        self.message_label.setStyleSheet("")
