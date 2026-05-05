"""Rubric category editor card widget."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.core.models import Category


class RubricCategoryCard(QWidget):
    """Editable card for a single rubric category."""

    changed = pyqtSignal()  # Emitted on any edit
    delete_requested = pyqtSignal(object)  # Emits self

    def __init__(
        self, category: Category | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("categoryCard")
        self._collapsed = False
        self._setup_ui()
        if category:
            self.load_category(category)

    def _setup_ui(self) -> None:
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(10, 8, 10, 8)
        self.layout_.setSpacing(6)

        # Header row: name + collapse + delete
        header = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Category name")
        self.name_edit.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.name_edit.textChanged.connect(self.changed.emit)
        header.addWidget(self.name_edit, 1)

        self.collapse_btn = QPushButton("^")
        self.collapse_btn.setFixedSize(28, 28)
        self.collapse_btn.clicked.connect(self._toggle_collapse)
        header.addWidget(self.collapse_btn)

        self.delete_btn = QPushButton("x")
        self.delete_btn.setObjectName("dangerButton")
        self.delete_btn.setFixedSize(28, 28)
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(self))
        header.addWidget(self.delete_btn)
        self.layout_.addLayout(header)

        # Collapsible body
        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(6)

        # Definition
        body_layout.addWidget(QLabel("Definition:"))
        self.definition_edit = QPlainTextEdit()
        self.definition_edit.setPlaceholderText("What this category means...")
        self.definition_edit.setMaximumHeight(80)
        self.definition_edit.textChanged.connect(self.changed.emit)
        body_layout.addWidget(self.definition_edit)

        # Exemplars
        exemplar_header = QHBoxLayout()
        self.exemplar_label = QLabel("Exemplars:")
        self.exemplar_label.setToolTip(
            "Verbatim chunk-like strings the LLM uses as reference. "
            "Drag to reorder, double-click to edit."
        )
        exemplar_header.addWidget(self.exemplar_label)
        exemplar_header.addStretch()
        add_ex_btn = QPushButton("+ Add")
        add_ex_btn.clicked.connect(self._add_exemplar)
        exemplar_header.addWidget(add_ex_btn)
        rm_ex_btn = QPushButton("Remove")
        rm_ex_btn.clicked.connect(self._remove_exemplar)
        exemplar_header.addWidget(rm_ex_btn)
        body_layout.addLayout(exemplar_header)

        self.exemplar_list = QListWidget()
        self.exemplar_list.setMinimumHeight(140)
        self.exemplar_list.setMaximumHeight(260)
        self.exemplar_list.setWordWrap(True)
        self.exemplar_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.exemplar_list.itemDoubleClicked.connect(self._edit_exemplar)
        body_layout.addWidget(self.exemplar_list)

        # Boundary cases — disambiguation rules for borderline chunks.
        bc_header = QHBoxLayout()
        self.boundary_label = QLabel("Boundary cases:")
        self.boundary_label.setToolTip(
            "Disambiguation notes for the LLM, e.g. 'If X but Y, prefer category Z'.\n"
            "Used to resolve borderline chunks against neighboring categories."
        )
        bc_header.addWidget(self.boundary_label)
        bc_header.addStretch()
        add_bc_btn = QPushButton("+ Add")
        add_bc_btn.clicked.connect(self._add_boundary_case)
        bc_header.addWidget(add_bc_btn)
        rm_bc_btn = QPushButton("Remove")
        rm_bc_btn.clicked.connect(self._remove_boundary_case)
        bc_header.addWidget(rm_bc_btn)
        body_layout.addLayout(bc_header)

        self.boundary_list = QListWidget()
        self.boundary_list.setMinimumHeight(80)
        self.boundary_list.setMaximumHeight(180)
        self.boundary_list.setWordWrap(True)
        self.boundary_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.boundary_list.itemDoubleClicked.connect(self._edit_boundary_case)
        body_layout.addWidget(self.boundary_list)

        # Confidence threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Confidence threshold:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(85)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("0.85")
        self.threshold_label.setFixedWidth(35)
        threshold_layout.addWidget(self.threshold_label)
        body_layout.addLayout(threshold_layout)

        self.layout_.addWidget(self.body)

    def load_category(self, cat: Category) -> None:
        """Load a Category into this card."""
        self.name_edit.setText(cat.name)
        self.definition_edit.setPlainText(cat.definition)

        self.exemplar_list.clear()
        for ex in cat.exemplars:
            self.exemplar_list.addItem(ex)
        self.exemplar_label.setText(f"Exemplars ({len(cat.exemplars)}):")

        self.boundary_list.clear()
        for bc in cat.boundary_cases or []:
            self.boundary_list.addItem(bc)
        self.boundary_label.setText(
            f"Boundary cases ({len(cat.boundary_cases or [])}):"
        )

        self.threshold_slider.setValue(int(cat.confidence_threshold * 100))

    def to_category(self) -> Category:
        """Extract current state as a Category."""
        exemplars = [
            self.exemplar_list.item(i).text()
            for i in range(self.exemplar_list.count())
            if self.exemplar_list.item(i)
        ]
        boundary_cases = [
            self.boundary_list.item(i).text()
            for i in range(self.boundary_list.count())
            if self.boundary_list.item(i)
        ]
        return Category(
            name=self.name_edit.text().strip(),
            definition=self.definition_edit.toPlainText().strip(),
            exemplars=exemplars,
            boundary_cases=boundary_cases,
            confidence_threshold=self.threshold_slider.value() / 100.0,
        )

    def _toggle_collapse(self) -> None:
        self._collapsed = not self._collapsed
        self.body.setVisible(not self._collapsed)
        self.collapse_btn.setText("v" if self._collapsed else "^")

    def _refresh_exemplar_count_label(self) -> None:
        self.exemplar_label.setText(f"Exemplars ({self.exemplar_list.count()}):")

    def _refresh_boundary_count_label(self) -> None:
        self.boundary_label.setText(
            f"Boundary cases ({self.boundary_list.count()}):"
        )

    def _add_exemplar(self) -> None:
        self.exemplar_list.addItem("(double-click to edit)")
        item = self.exemplar_list.item(self.exemplar_list.count() - 1)
        if item:
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.exemplar_list.editItem(item)
        self._refresh_exemplar_count_label()
        self.changed.emit()

    def _edit_exemplar(self, item) -> None:
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.exemplar_list.editItem(item)
        self.changed.emit()

    def _remove_exemplar(self) -> None:
        current = self.exemplar_list.currentRow()
        if current >= 0:
            self.exemplar_list.takeItem(current)
            self._refresh_exemplar_count_label()
            self.changed.emit()

    def _add_boundary_case(self) -> None:
        self.boundary_list.addItem(
            "(double-click to edit, e.g. 'If X but Y, prefer category Z')"
        )
        item = self.boundary_list.item(self.boundary_list.count() - 1)
        if item:
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.boundary_list.editItem(item)
        self._refresh_boundary_count_label()
        self.changed.emit()

    def _edit_boundary_case(self, item) -> None:
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.boundary_list.editItem(item)
        self.changed.emit()

    def _remove_boundary_case(self) -> None:
        current = self.boundary_list.currentRow()
        if current >= 0:
            self.boundary_list.takeItem(current)
            self._refresh_boundary_count_label()
            self.changed.emit()

    def _on_threshold_changed(self, value: int) -> None:
        self.threshold_label.setText(f"{value / 100:.2f}")
        self.changed.emit()
