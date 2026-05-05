"""Left panel: document tree with status indicators."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class LeftPanel(QWidget):
    """Document list panel with status indicators."""

    document_selected = pyqtSignal(int)  # doc_id
    open_folder_requested = pyqtSignal()
    add_documents_requested = pyqtSignal()
    reset_project_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header = QLabel("Documents")
        header.setObjectName("sectionHeader")
        layout.addWidget(header)

        # Document tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Status", "Chunks"])
        self.tree.setColumnCount(3)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.tree.header().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.currentItemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree)

        # Buttons
        btn_layout = QHBoxLayout()
        self.open_btn = QPushButton("Open Folder")
        self.open_btn.setObjectName("accentButton")
        self.open_btn.clicked.connect(self.open_folder_requested.emit)
        btn_layout.addWidget(self.open_btn)

        self.add_btn = QPushButton("Add Files")
        self.add_btn.clicked.connect(self.add_documents_requested.emit)
        btn_layout.addWidget(self.add_btn)
        layout.addLayout(btn_layout)

        # Count label
        self.count_label = QLabel("0 documents")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.count_label)

        # Reset project (destructive — wipes project.db)
        self.reset_btn = QPushButton("Reset Project")
        self.reset_btn.setObjectName("dangerButton")
        self.reset_btn.setToolTip(
            "Delete project.db: every chunk, label, review, and audit event. "
            "The source documents on disk are kept."
        )
        self.reset_btn.clicked.connect(self.reset_project_requested.emit)
        layout.addWidget(self.reset_btn)

    def populate(self, documents: list) -> None:
        """Populate tree from list of ConvertedDocument."""
        self.tree.clear()
        for doc in documents:
            item = QTreeWidgetItem()
            item.setText(0, doc.filename)
            item.setData(0, Qt.ItemDataRole.UserRole, doc.id)

            # Status indicator
            status = doc.status
            item.setText(1, status)
            if status == "parsed":
                item.setForeground(1, QColor("#51cf66"))
            elif status == "warnings":
                item.setForeground(1, QColor("#ffd43b"))
            elif status == "failed":
                item.setForeground(1, QColor("#ff6b6b"))
            else:
                item.setForeground(1, QColor("#888"))

            item.setText(2, "")  # Chunk count filled later
            self.tree.addTopLevelItem(item)

        self.count_label.setText(f"{len(documents)} documents")

    def update_chunk_count(self, doc_id: int, count: int) -> None:
        """Update the chunk count for a document."""
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item and item.data(0, Qt.ItemDataRole.UserRole) == doc_id:
                item.setText(2, str(count))
                break

    def select_document(self, doc_id: int) -> None:
        """Programmatically select a document."""
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item and item.data(0, Qt.ItemDataRole.UserRole) == doc_id:
                self.tree.setCurrentItem(item)
                break

    def _on_item_changed(
        self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None
    ) -> None:
        if current is not None:
            doc_id = current.data(0, Qt.ItemDataRole.UserRole)
            if doc_id is not None:
                self.document_selected.emit(doc_id)
