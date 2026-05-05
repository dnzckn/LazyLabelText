"""Results mode: in-GUI results review with statistics and filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.ui.modes.base_mode import BaseMode
from lazylabeltext.ui.widgets.confidence_bar import ConfidenceBar

if TYPE_CHECKING:
    from lazylabeltext.core.app_context import AppContext


class ResultsModeWidget(BaseMode):
    """Dashboard and filterable table for reviewing labeling results."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(context, parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: dashboard stats
        dashboard = QWidget()
        dash_layout = QHBoxLayout(dashboard)
        dash_layout.setContentsMargins(12, 8, 12, 8)

        self.stat_widgets: dict[str, QLabel] = {}
        for name in ["Total Chunks", "Labeled", "Reviewed", "Avg Confidence"]:
            stat = QWidget()
            stat_layout = QVBoxLayout(stat)
            stat_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label = QLabel("0")
            value_label.setObjectName("statValue")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stat_layout.addWidget(value_label)
            name_label = QLabel(name)
            name_label.setObjectName("statLabel")
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stat_layout.addWidget(name_label)
            dash_layout.addWidget(stat)
            self.stat_widgets[name] = value_label

        splitter.addWidget(dashboard)

        # Bottom: filter + table + detail
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(8, 4, 8, 4)

        # Filters
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))

        self.doc_filter = QComboBox()
        self.doc_filter.addItem("All Documents")
        self.doc_filter.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.doc_filter)

        self.cat_filter = QComboBox()
        self.cat_filter.addItem("All Categories")
        self.cat_filter.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.cat_filter)

        self.status_filter = QComboBox()
        self.status_filter.addItems(["All Status", "Unreviewed", "Accepted", "Flagged"])
        self.status_filter.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.status_filter)

        filter_row.addStretch()
        bottom_layout.addLayout(filter_row)

        # Results table
        inner_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Text", "Document", "Category", "Confidence", "Status"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.currentCellChanged.connect(self._on_row_selected)
        inner_splitter.addWidget(self.table)

        # Detail panel
        detail = QWidget()
        detail.setMaximumWidth(350)
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(4, 4, 4, 4)

        detail_header = QLabel("Details")
        detail_header.setObjectName("sectionHeader")
        detail_layout.addWidget(detail_header)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setPlaceholderText("Select a row to see details.")
        detail_layout.addWidget(self.detail_text)

        self.detail_confidence = QWidget()
        self.detail_confidence_layout = QVBoxLayout(self.detail_confidence)
        self.detail_confidence_layout.setSpacing(4)
        detail_layout.addWidget(self.detail_confidence)

        self.detail_rationale = QLabel()
        self.detail_rationale.setWordWrap(True)
        self.detail_rationale.setStyleSheet(
            "color: #aaa; font-style: italic; font-size: 11px;"
        )
        detail_layout.addWidget(self.detail_rationale)

        inner_splitter.addWidget(detail)
        inner_splitter.setSizes([600, 300])

        bottom_layout.addWidget(inner_splitter, 1)
        splitter.addWidget(bottom)

        splitter.setSizes([80, 500])
        layout.addWidget(splitter)

    def activate(self) -> None:
        self._load_results()

    def _load_results(self) -> None:
        if self.ctx.rubric_manager is None or self.ctx.database is None:
            return

        rubric = self.ctx.rubric_manager.get_active_rubric()
        if rubric is None:
            return

        summary = self.ctx.database.get_labeling_summary(rubric.id)

        # Update stats
        self.stat_widgets["Total Chunks"].setText(str(summary.get("total_labels", 0)))
        self.stat_widgets["Labeled"].setText(str(summary.get("total_labels", 0)))
        self.stat_widgets["Reviewed"].setText(str(summary.get("reviewed", 0)))

        confidences = summary.get("confidences", [])
        avg = sum(confidences) / len(confidences) if confidences else 0
        self.stat_widgets["Avg Confidence"].setText(f"{avg:.0%}")

        # Populate filters
        self.doc_filter.clear()
        self.doc_filter.addItem("All Documents")
        if self.ctx.document_manager:
            for doc in self.ctx.document_manager.get_all_documents():
                self.doc_filter.addItem(doc.filename, doc.id)

        self.cat_filter.clear()
        self.cat_filter.addItem("All Categories")
        for cat_name in summary.get("category_counts", {}):
            self.cat_filter.addItem(cat_name)

        # Populate table
        self._populate_table(rubric.id)

    def _populate_table(self, rubric_version_id: int) -> None:
        if self.ctx.database is None:
            return

        labels = self.ctx.database.get_all_labels(rubric_version_id)
        self.table.setRowCount(len(labels))
        self._label_data = labels

        for row, label in enumerate(labels):
            chunk = self.ctx.database.get_chunk(label.chunk_id)
            doc = self.ctx.database.get_document(chunk.document_id) if chunk else None
            reviews = self.ctx.database.get_reviews_for_label(label.id or 0)
            review = reviews[-1] if reviews else None

            # Text preview
            text = (
                chunk.text[:80] + "..."
                if chunk and len(chunk.text) > 80
                else (chunk.text if chunk else "")
            )
            self.table.setItem(row, 0, QTableWidgetItem(text))

            # Document — store doc_id on the item so filtering can match by id
            doc_name = doc.filename if doc else "?"
            doc_item = QTableWidgetItem(doc_name)
            if doc and doc.id is not None:
                doc_item.setData(Qt.ItemDataRole.UserRole, doc.id)
            self.table.setItem(row, 1, doc_item)

            # Category — prefer the human-corrected categories when present
            if review and review.final_categories:
                cats = ", ".join(review.final_categories)
            else:
                cats = ", ".join(label.predicted_categories)
            self.table.setItem(row, 2, QTableWidgetItem(cats))

            # Confidence — human review trumps LLM confidence with implicit 1.0
            if review and review.action in ("accept", "correct"):
                conf_text = "100%"
            else:
                conf_text = f"{label.composite_confidence:.0%}"
            self.table.setItem(row, 3, QTableWidgetItem(conf_text))

            # Status
            status = review.action if review else "unreviewed"
            self.table.setItem(row, 4, QTableWidgetItem(status))

    def _apply_filters(self) -> None:
        """Filter table rows based on current filter selections."""
        doc_filter_id = self.doc_filter.currentData()  # int doc_id or None
        cat_filter = self.cat_filter.currentText()
        status_filter = self.status_filter.currentText()

        # Map UI status labels to the action strings stored in human_reviews.
        status_map = {
            "Accepted": "accept",
            "Flagged": "flag",
            "Unreviewed": "unreviewed",
        }

        for row in range(self.table.rowCount()):
            show = True

            if doc_filter_id is not None:
                doc_item = self.table.item(row, 1)
                row_doc_id = (
                    doc_item.data(Qt.ItemDataRole.UserRole) if doc_item else None
                )
                if row_doc_id != doc_filter_id:
                    show = False

            if show and cat_filter != "All Categories":
                cat_item = self.table.item(row, 2)
                if cat_item and cat_filter not in cat_item.text():
                    show = False

            if show and status_filter != "All Status":
                expected = status_map.get(status_filter, status_filter.lower())
                status_item = self.table.item(row, 4)
                if status_item and status_item.text().lower() != expected.lower():
                    show = False

            self.table.setRowHidden(row, not show)

    def _on_row_selected(self, row: int, *_args) -> None:
        if row < 0 or row >= len(self._label_data):
            return

        label = self._label_data[row]
        chunk = (
            self.ctx.database.get_chunk(label.chunk_id) if self.ctx.database else None
        )

        if chunk:
            self.detail_text.setPlainText(chunk.text)

        # Confidence bars
        while self.detail_confidence_layout.count():
            child = self.detail_confidence_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for cat, conf in sorted(
            label.confidence_per_category.items(), key=lambda x: -x[1]
        ):
            bar = ConfidenceBar(value=conf, label=cat)
            self.detail_confidence_layout.addWidget(bar)

        self.detail_rationale.setText(f"Rationale: {label.rationale}")
