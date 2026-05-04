"""Confidence bar widget with color gradient."""

from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QWidget


class ConfidenceBar(QWidget):
    """Horizontal bar showing confidence level with color gradient."""

    def __init__(
        self,
        value: float = 0.0,
        label: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._value = max(0.0, min(1.0, value))
        self._label = label
        self.setFixedHeight(24)
        self.setMinimumWidth(120)

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(1.0, value))
        self.update()

    def set_label(self, label: str) -> None:
        self._label = label
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(50, 50, 55))
        painter.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)

        # Filled portion
        if self._value > 0:
            color = self._get_color(self._value)
            painter.setBrush(color)
            fill_width = max(8, w * self._value)
            painter.drawRoundedRect(QRectF(0, 0, fill_width, h), 4, 4)

        # Text
        painter.setPen(QColor(255, 255, 255))
        text = (
            f"{self._label}  {self._value:.0%}" if self._label else f"{self._value:.0%}"
        )
        painter.drawText(QRectF(6, 0, w - 12, h), Qt.AlignmentFlag.AlignVCenter, text)

        painter.end()

    @staticmethod
    def _get_color(value: float) -> QColor:
        """Map confidence value to red-yellow-green gradient."""
        if value < 0.3:
            return QColor(220, 80, 80, 200)
        elif value < 0.7:
            r = int(220 + (200 - 220) * (value - 0.3) / 0.4)
            g = int(80 + (180 - 80) * (value - 0.3) / 0.4)
            return QColor(r, g, 60, 200)
        else:
            return QColor(80, 180, 80, 200)
