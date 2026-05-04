"""Center panel: QStackedWidget hosting mode-specific views."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget


class CenterPanel(QStackedWidget):
    """Hosts mode-specific views, one per mode."""

    MODE_NAMES = [
        "convert",
        "rubric",
        "chunk",
        "label",
        "results",
        "propagation",
        "export",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mode_widgets: dict[str, QWidget] = {}

        # Create placeholder pages for each mode
        for name in self.MODE_NAMES:
            placeholder = QWidget()
            layout = QVBoxLayout(placeholder)
            label = QLabel(f"{name.title()} mode")
            label.setStyleSheet("color: #666; font-size: 14px;")
            layout.addWidget(label)
            layout.addStretch()
            self._mode_widgets[name] = placeholder
            self.addWidget(placeholder)

    def set_mode_widget(self, mode: str, widget: QWidget) -> None:
        """Replace the placeholder for a mode with a real widget."""
        if mode not in self._mode_widgets:
            return
        old = self._mode_widgets[mode]
        idx = self.indexOf(old)
        self.removeWidget(old)
        old.deleteLater()
        self.insertWidget(idx, widget)
        self._mode_widgets[mode] = widget

    def set_mode(self, mode: str) -> None:
        """Switch to the specified mode's view."""
        widget = self._mode_widgets.get(mode)
        if widget:
            self.setCurrentWidget(widget)

    def get_mode_widget(self, mode: str) -> QWidget | None:
        return self._mode_widgets.get(mode)

    @property
    def current_mode(self) -> str:
        idx = self.currentIndex()
        if 0 <= idx < len(self.MODE_NAMES):
            return self.MODE_NAMES[idx]
        return "convert"
