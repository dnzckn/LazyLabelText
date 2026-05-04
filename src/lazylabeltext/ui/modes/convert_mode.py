"""Convert mode: side-by-side original and converted text view."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.ui.modes.base_mode import BaseMode

if TYPE_CHECKING:
    from lazylabeltext.core.app_context import AppContext


class ConvertModeWidget(BaseMode):
    """Shows original document alongside converted structured text."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: original text
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 4, 4, 4)
        left_header = QLabel("Source Text")
        left_header.setObjectName("sectionHeader")
        left_layout.addWidget(left_header)
        self.original_view = QTextEdit()
        self.original_view.setReadOnly(True)
        self.original_view.setPlaceholderText(
            "Select a document from the left panel to view its contents."
        )
        left_layout.addWidget(self.original_view)
        splitter.addWidget(left)

        # Right: converted text with heading highlights
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 8, 4)
        right_header = QLabel("Converted (with structure)")
        right_header.setObjectName("sectionHeader")
        right_layout.addWidget(right_header)
        self.converted_view = QTextEdit()
        self.converted_view.setReadOnly(True)
        self.converted_view.setPlaceholderText("Converted text will appear here.")
        right_layout.addWidget(self.converted_view)

        # Info bar
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #888; font-size: 11px; padding: 2px;")
        right_layout.addWidget(self.info_label)
        splitter.addWidget(right)

        splitter.setSizes([400, 400])
        layout.addWidget(splitter)

    def on_document_selected(self, doc_id: int) -> None:
        if self.ctx.document_manager is None:
            return

        doc = self.ctx.document_manager.get_document(doc_id)
        if doc is None:
            return

        self.original_view.setPlainText(doc.full_text)

        # Build converted view with highlighted headings
        html_parts = []
        heading_colors = {
            1: "#5c8fbf",
            2: "#7faf5c",
            3: "#bf8f5c",
            4: "#8f5cbf",
            5: "#5cbfbf",
            6: "#bf5c5c",
        }

        text = doc.full_text
        last_end = 0

        for heading in sorted(doc.headings, key=lambda h: h.char_start):
            # Add text before heading
            if heading.char_start > last_end:
                before = text[last_end : heading.char_start]
                html_parts.append(
                    f"<span style='color: #ccc;'>{self._escape(before)}</span>"
                )

            # Add heading with color
            color = heading_colors.get(heading.level, "#aaa")
            heading_text = text[heading.char_start : heading.char_end]
            size = max(12, 20 - heading.level * 2)
            html_parts.append(
                f"<div style='color: {color}; font-size: {size}px; font-weight: bold; "
                f"margin: 4px 0;'>"
                f"{'#' * heading.level} {self._escape(heading_text)}</div>"
            )
            last_end = heading.char_end

        # Remaining text
        if last_end < len(text):
            html_parts.append(
                f"<span style='color: #ccc;'>{self._escape(text[last_end:])}</span>"
            )

        self.converted_view.setHtml(
            "<div style='white-space: pre-wrap; font-family: monospace;'>"
            + "".join(html_parts)
            + "</div>"
        )

        # Info
        info = (
            f"{len(doc.full_text):,} chars | "
            f"{len(doc.headings)} headings | "
            f"{len(doc.pages)} pages | "
            f"Format: {doc.format}"
        )
        if doc.warnings:
            info += f" | Warnings: {len(doc.warnings)}"
        self.info_label.setText(info)

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
