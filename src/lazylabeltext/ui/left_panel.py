"""Left panel: document tree with status indicators."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class LeftPanel(QWidget):
    """Document list panel with status indicators and per-doc context actions."""

    document_selected = pyqtSignal(int)  # doc_id
    open_folder_requested = pyqtSignal()
    open_docmap_requested = pyqtSignal()
    add_documents_requested = pyqtSignal()
    reset_project_requested = pyqtSignal()
    document_clear_requested = pyqtSignal(int)  # wipe chunks+labels for one doc
    document_delete_requested = pyqtSignal(int)  # remove doc entirely from project

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QLabel("Documents")
        header.setObjectName("sectionHeader")
        layout.addWidget(header)

        # Document tree — Name | Chunks | Labels | Reviewed
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Chunks", "Labels", "Reviewed"])
        self.tree.setColumnCount(4)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 4):
            self.tree.header().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.currentItemChanged.connect(self._on_item_changed)

        # Right-click context menu
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.tree)

        btn_layout = QHBoxLayout()
        self.open_btn = QPushButton("Open Folder")
        self.open_btn.setObjectName("accentButton")
        self.open_btn.clicked.connect(self.open_folder_requested.emit)
        btn_layout.addWidget(self.open_btn)

        self.open_docmap_btn = QPushButton("Open Doc Map")
        self.open_docmap_btn.setToolTip(
            "Open a project from a doc-map file (paths/globs across disk) "
            "instead of a single folder. Source documents stay where they are."
        )
        self.open_docmap_btn.clicked.connect(self.open_docmap_requested.emit)
        btn_layout.addWidget(self.open_docmap_btn)

        self.add_btn = QPushButton("Add Files")
        self.add_btn.clicked.connect(self.add_documents_requested.emit)
        btn_layout.addWidget(self.add_btn)
        layout.addLayout(btn_layout)

        self.count_label = QLabel("0 documents")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.count_label)

        self.reset_btn = QPushButton("Reset Project")
        self.reset_btn.setObjectName("dangerButton")
        self.reset_btn.setToolTip(
            "Delete project.db: every chunk, label, review, and audit event. "
            "The source documents on disk are kept."
        )
        self.reset_btn.clicked.connect(self.reset_project_requested.emit)
        layout.addWidget(self.reset_btn)

    # --- Population ----------------------------------------------------

    def populate(self, documents: list) -> None:
        """Populate tree from list of ConvertedDocument (no stats yet)."""
        self.tree.clear()
        for doc in documents:
            item = QTreeWidgetItem()
            item.setText(0, doc.filename)
            item.setData(0, Qt.ItemDataRole.UserRole, doc.id)
            item.setData(0, Qt.ItemDataRole.UserRole + 1, doc.status)
            self._apply_status_color(item, doc.status, chunks=0, labels=0, reviewed=0)
            item.setText(1, "")
            item.setText(2, "")
            item.setText(3, "")
            self.tree.addTopLevelItem(item)
        self.count_label.setText(f"{len(documents)} documents")

    def update_chunk_count(self, doc_id: int, count: int) -> None:
        """Update only the chunk count column for a document."""
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item and item.data(0, Qt.ItemDataRole.UserRole) == doc_id:
                item.setText(1, str(count))
                self._recolor_row(item)
                break

    def update_document_stats(self, stats: list[dict]) -> None:
        """Refresh all per-doc columns from `Database.get_document_label_status()`.

        Each entry: {filename, status, chunk_count, label_count, reviewed_count}.
        Missing keys default to 0.
        """
        # Build a lookup so we can match by id when present, else by filename.
        by_id: dict[int, dict] = {}
        by_name: dict[str, dict] = {}
        for s in stats:
            if "document_id" in s and s["document_id"] is not None:
                by_id[s["document_id"]] = s
            if "filename" in s:
                by_name[s["filename"]] = s

        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if not item:
                continue
            doc_id = item.data(0, Qt.ItemDataRole.UserRole)
            entry = by_id.get(doc_id) or by_name.get(item.text(0))
            if entry is None:
                continue
            chunks = int(entry.get("chunk_count", 0))
            labels = int(entry.get("label_count", 0))
            reviewed = int(entry.get("reviewed_count", 0))
            item.setText(1, str(chunks))
            item.setText(
                2,
                f"{labels}/{chunks}" if chunks else str(labels),
            )
            item.setText(
                3,
                f"{reviewed}/{labels}" if labels else str(reviewed),
            )
            doc_status = item.data(0, Qt.ItemDataRole.UserRole + 1) or "parsed"
            self._apply_status_color(
                item, doc_status, chunks=chunks, labels=labels, reviewed=reviewed
            )

    def _apply_status_color(
        self,
        item: QTreeWidgetItem,
        status: str,
        chunks: int,
        labels: int,
        reviewed: int,
    ) -> None:
        """Color the row by labeling progress (and parse status as a fallback)."""
        if status == "failed":
            color = QColor("#ff6b6b")  # red — couldn't parse
        elif status == "warnings":
            color = QColor("#ffd43b")
        elif chunks == 0:
            color = QColor("#888")  # grey — not chunked yet
        elif labels == 0:
            color = QColor("#bbbbbb")  # light grey — chunked, unlabeled
        elif labels < chunks:
            color = QColor("#4dabf7")  # blue — partially labeled
        elif reviewed < labels:
            color = QColor("#ffd43b")  # yellow — fully labeled, not all reviewed
        else:
            color = QColor("#51cf66")  # green — labeled and reviewed
        for col in range(item.columnCount()):
            item.setForeground(col, color)

    def _recolor_row(self, item: QTreeWidgetItem) -> None:
        """Re-derive color from current cell values (for partial updates)."""
        try:
            chunks = int(item.text(1) or 0)
        except ValueError:
            chunks = 0
        labels_text = item.text(2)
        labels = 0
        if "/" in labels_text:
            try:
                labels = int(labels_text.split("/", 1)[0])
            except ValueError:
                labels = 0
        else:
            try:
                labels = int(labels_text or 0)
            except ValueError:
                labels = 0
        reviewed_text = item.text(3)
        reviewed = 0
        if "/" in reviewed_text:
            try:
                reviewed = int(reviewed_text.split("/", 1)[0])
            except ValueError:
                reviewed = 0
        doc_status = item.data(0, Qt.ItemDataRole.UserRole + 1) or "parsed"
        self._apply_status_color(item, doc_status, chunks, labels, reviewed)

    def select_document(self, doc_id: int) -> None:
        """Programmatically select a document."""
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item and item.data(0, Qt.ItemDataRole.UserRole) == doc_id:
                self.tree.setCurrentItem(item)
                break

    # --- Signals -------------------------------------------------------

    def _on_item_changed(
        self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None
    ) -> None:
        if current is not None:
            doc_id = current.data(0, Qt.ItemDataRole.UserRole)
            if doc_id is not None:
                self.document_selected.emit(doc_id)

    def _on_context_menu(self, position) -> None:
        item = self.tree.itemAt(position)
        if item is None:
            return
        doc_id = item.data(0, Qt.ItemDataRole.UserRole)
        if doc_id is None:
            return
        filename = item.text(0)

        menu = QMenu(self)
        clear_act = QAction("Clear chunks + labels for this document", self)
        clear_act.setToolTip(
            f"Wipe all chunks, labels, reviews, and chunking runs for "
            f"{filename}. The document itself stays in the project."
        )
        clear_act.triggered.connect(
            lambda: self.document_clear_requested.emit(doc_id)
        )
        menu.addAction(clear_act)

        delete_act = QAction("Remove document from project", self)
        delete_act.setObjectName("dangerAction")
        delete_act.setToolTip(
            f"Remove {filename} from the project entirely (including its "
            f"chunks, labels, and reviews). The source file on disk is kept."
        )
        delete_act.triggered.connect(
            lambda: self.document_delete_requested.emit(doc_id)
        )
        menu.addAction(delete_act)

        menu.exec(self.tree.viewport().mapToGlobal(position))
