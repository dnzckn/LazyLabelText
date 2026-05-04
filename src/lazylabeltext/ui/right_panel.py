"""Right panel: rubric summary and context info."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.core.models import Rubric


class RightPanel(QWidget):
    """Rubric summary and contextual information panel."""

    rubric_edit_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Header
        header = QLabel("Rubric")
        header.setObjectName("sectionHeader")
        layout.addWidget(header)

        # Version indicator
        self.version_label = QLabel("No rubric loaded")
        self.version_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.version_label)

        # Scrollable category list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.category_container = QWidget()
        self.category_layout = QVBoxLayout(self.category_container)
        self.category_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.category_layout.setSpacing(4)
        scroll.setWidget(self.category_container)
        layout.addWidget(scroll, 1)

        # Edit button
        self.edit_btn = QPushButton("Edit Rubric")
        self.edit_btn.setObjectName("accentButton")
        self.edit_btn.clicked.connect(self.rubric_edit_requested.emit)
        layout.addWidget(self.edit_btn)

    def update_rubric(self, rubric: Rubric | None) -> None:
        """Update the rubric display."""
        # Clear existing categories
        while self.category_layout.count():
            child = self.category_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if rubric is None:
            self.version_label.setText("No rubric loaded")
            return

        self.version_label.setText(f"{rubric.name} v{rubric.version}")

        for cat in rubric.categories:
            card = QWidget()
            card.setObjectName("categoryCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 6, 8, 6)
            card_layout.setSpacing(2)

            name_label = QLabel(f"<b>{cat.name}</b>")
            card_layout.addWidget(name_label)

            if cat.definition:
                def_label = QLabel(cat.definition)
                def_label.setWordWrap(True)
                def_label.setStyleSheet("color: #aaa; font-size: 11px;")
                card_layout.addWidget(def_label)

            if cat.exemplars:
                ex_label = QLabel(f"{len(cat.exemplars)} exemplars")
                ex_label.setStyleSheet("color: #888; font-size: 10px;")
                card_layout.addWidget(ex_label)

            self.category_layout.addWidget(card)
