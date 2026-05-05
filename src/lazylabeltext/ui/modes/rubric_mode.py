"""Rubric mode: structured category editor with corpus-wide coverage map."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.core.models import Category
from lazylabeltext.ui.modes.base_mode import BaseMode
from lazylabeltext.ui.widgets.rubric_category_card import RubricCategoryCard

if TYPE_CHECKING:
    from lazylabeltext.core.app_context import AppContext
    from lazylabeltext.ui.main_window import MainWindow

logger = logging.getLogger("lazylabeltext")


class RubricModeWidget(BaseMode):
    """Rubric editor with a sandbox area that classifies sample chunks using the in-memory rubric."""

    def __init__(
        self,
        context: AppContext,
        main_window: MainWindow,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(context, parent)
        self.main_window = main_window
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 4, 8, 4)

        self.save_btn = QPushButton("Save Rubric")
        self.save_btn.setObjectName("accentButton")
        self.save_btn.clicked.connect(self._save_rubric)
        toolbar.addWidget(self.save_btn)

        import_btn = QPushButton("Import JSON")
        import_btn.clicked.connect(self._import_json)
        toolbar.addWidget(import_btn)

        export_btn = QPushButton("Export JSON")
        export_btn.clicked.connect(self._export_json)
        toolbar.addWidget(export_btn)

        toolbar.addStretch()

        self.version_label = QLabel()
        self.version_label.setStyleSheet("color: #888;")
        toolbar.addWidget(self.version_label)

        layout.addLayout(toolbar)

        # Main content: editor | preview
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: category cards
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 4, 4, 4)

        left_header = QLabel("Categories")
        left_header.setObjectName("sectionHeader")
        left_layout.addWidget(left_header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.cards_layout.setSpacing(8)
        scroll.setWidget(self.cards_container)
        left_layout.addWidget(scroll, 1)

        add_btn = QPushButton("+ Add Category")
        add_btn.setObjectName("accentButton")
        add_btn.clicked.connect(self._add_category)
        left_layout.addWidget(add_btn)

        splitter.addWidget(left)

        # Right: corpus-wide coverage map for the active rubric
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 8, 4)

        coverage_header_row = QHBoxLayout()
        self.coverage_header = QLabel("Coverage")
        self.coverage_header.setObjectName("sectionHeader")
        coverage_header_row.addWidget(self.coverage_header)
        coverage_header_row.addStretch()
        self.coverage_refresh_btn = QPushButton("Refresh")
        self.coverage_refresh_btn.clicked.connect(self._refresh_coverage)
        coverage_header_row.addWidget(self.coverage_refresh_btn)
        right_layout.addLayout(coverage_header_row)

        self.coverage_status_label = QLabel(
            "Per-category corpus health for the saved rubric. "
            "Refreshes when you switch to this tab or click Refresh."
        )
        self.coverage_status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.coverage_status_label.setWordWrap(True)
        right_layout.addWidget(self.coverage_status_label)

        self.coverage_table = QTableWidget()
        self.coverage_table.setColumnCount(5)
        self.coverage_table.setHorizontalHeaderLabels(
            ["Category", "Count", "Avg Conf", "Docs", "Disagree %"]
        )
        self.coverage_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for i in range(1, 5):
            self.coverage_table.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.ResizeMode.ResizeToContents
            )
        self.coverage_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.coverage_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        self.coverage_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.coverage_table.currentCellChanged.connect(self._on_coverage_row_selected)
        right_layout.addWidget(self.coverage_table, 1)

        self.coverage_detail = QTextEdit()
        self.coverage_detail.setReadOnly(True)
        self.coverage_detail.setMaximumHeight(180)
        self.coverage_detail.setPlaceholderText(
            "Select a category to see its definition and any health flags."
        )
        right_layout.addWidget(self.coverage_detail)

        splitter.addWidget(right)
        splitter.setSizes([400, 500])

        layout.addWidget(splitter, 1)

        self._coverage_rows: list[dict] = []

    def activate(self) -> None:
        """Load current rubric into editor."""
        if self.ctx.rubric_manager is None:
            return

        rubric = self.ctx.rubric_manager.get_active_rubric()
        self._clear_cards()

        if rubric:
            self.version_label.setText(f"v{rubric.version}")
            for cat in rubric.categories:
                self._add_category_card(cat)
        else:
            self.version_label.setText("No rubric")

        self._refresh_coverage()

    def _add_category(self) -> None:
        self._add_category_card(Category(name="", definition=""))

    def _add_category_card(self, category: Category) -> None:
        card = RubricCategoryCard(category)
        card.changed.connect(self._on_rubric_changed)
        card.delete_requested.connect(self._remove_category)
        self.cards_layout.addWidget(card)

    def _remove_category(self, card: RubricCategoryCard) -> None:
        self.cards_layout.removeWidget(card)
        card.deleteLater()
        self._on_rubric_changed()

    def _clear_cards(self) -> None:
        while self.cards_layout.count():
            child = self.cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _get_categories(self) -> list[Category]:
        categories = []
        for i in range(self.cards_layout.count()):
            widget = self.cards_layout.itemAt(i).widget()
            if isinstance(widget, RubricCategoryCard):
                cat = widget.to_category()
                if cat.name.strip():
                    categories.append(cat)
        return categories

    def _on_rubric_changed(self) -> None:
        """Edits don't affect coverage (which reflects saved rubric + labels)."""
        return

    # --- Coverage map ----------------------------------------------------

    def _refresh_coverage(self) -> None:
        if self.ctx.label_manager is None or self.ctx.rubric_manager is None:
            self._coverage_rows = []
            self.coverage_table.setRowCount(0)
            self.coverage_status_label.setText("Open a project first.")
            return

        rubric = self.ctx.rubric_manager.get_active_rubric()
        if rubric is None:
            self._coverage_rows = []
            self.coverage_table.setRowCount(0)
            self.coverage_status_label.setText("Save a rubric first.")
            return

        rows = self.ctx.label_manager.get_category_coverage(rubric.id)
        self._coverage_rows = rows
        self.coverage_table.setRowCount(len(rows))

        total_labels = sum(r["count"] for r in rows)
        no_labels_yet = total_labels == 0

        warn_dead = 0
        warn_fuzzy = 0
        warn_disagree = 0
        warn_orphan = 0

        for i, r in enumerate(rows):
            name = r["name"]
            count = r["count"]
            avg_conf = r["avg_confidence"]
            docs = r["doc_count"]
            disagree = r["disagree_pct"]

            name_item = QTableWidgetItem(name + ("" if r["in_rubric"] else "  (orphan)"))
            count_item = QTableWidgetItem(str(count))
            conf_item = QTableWidgetItem(f"{avg_conf:.0%}" if count else "—")
            docs_item = QTableWidgetItem(str(docs))
            dis_item = QTableWidgetItem(
                f"{disagree:.0%}" if r["review_count"] else "—"
            )

            # Health colouring per row.
            color: QColor | None = None
            flags: list[str] = []
            if not r["in_rubric"]:
                color = QColor(160, 160, 160)
                flags.append("orphan (not in active rubric)")
                warn_orphan += 1
            elif count == 0:
                # Pre-labeling: don't flag as "dead", just show as not-yet-used.
                if no_labels_yet:
                    color = QColor(160, 160, 160)
                else:
                    color = QColor(220, 120, 120)
                    flags.append("dead — no chunks labeled with this category")
                    warn_dead += 1
            else:
                if avg_conf < 0.6:
                    color = QColor(220, 200, 120)
                    flags.append(f"avg confidence is low ({avg_conf:.0%})")
                    warn_fuzzy += 1
                if r["review_count"] >= 3 and disagree >= 0.3:
                    color = QColor(220, 200, 120)
                    flags.append(
                        f"humans corrected {disagree:.0%} of reviews "
                        f"({r['review_count']} sampled)"
                    )
                    warn_disagree += 1

            if color is not None:
                brush = QBrush(color)
                for it in (name_item, count_item, conf_item, docs_item, dis_item):
                    it.setForeground(brush)

            # Stash flags + the category itself for the detail panel.
            name_item.setData(Qt.ItemDataRole.UserRole, flags)

            self.coverage_table.setItem(i, 0, name_item)
            self.coverage_table.setItem(i, 1, count_item)
            self.coverage_table.setItem(i, 2, conf_item)
            self.coverage_table.setItem(i, 3, docs_item)
            self.coverage_table.setItem(i, 4, dis_item)

        # Header status with summary.
        if no_labels_yet:
            self.coverage_status_label.setText(
                f"Saved: v{rubric.version}  ·  No labels yet — run labeling on the Label "
                "tab to populate coverage stats. Definitions are shown below for review."
            )
        else:
            bits: list[str] = [
                f"Saved: v{rubric.version}",
                f"{total_labels} labels across {len(rows)} categories",
            ]
            if warn_dead:
                bits.append(f"{warn_dead} dead")
            if warn_fuzzy:
                bits.append(f"{warn_fuzzy} low-confidence")
            if warn_disagree:
                bits.append(f"{warn_disagree} high-disagreement")
            if warn_orphan:
                bits.append(f"{warn_orphan} orphan")
            if not (warn_dead or warn_fuzzy or warn_disagree or warn_orphan):
                bits.append("all categories look healthy")
            self.coverage_status_label.setText("  ·  ".join(bits))

        if self._coverage_rows and self.coverage_table.currentRow() < 0:
            self.coverage_table.selectRow(0)

    def _on_coverage_row_selected(self, row: int, *_args) -> None:
        if row < 0 or row >= len(self._coverage_rows):
            self.coverage_detail.clear()
            return

        entry = self._coverage_rows[row]
        rubric = (
            self.ctx.rubric_manager.get_active_rubric()
            if self.ctx.rubric_manager
            else None
        )
        cat: Category | None = None
        if rubric:
            for c in rubric.categories:
                if c.name == entry["name"]:
                    cat = c
                    break

        flags = self.coverage_table.item(row, 0).data(Qt.ItemDataRole.UserRole) or []

        lines: list[str] = [f"Category: {entry['name']}"]
        if not entry["in_rubric"]:
            lines.append("(orphan: this category appears on labels but is not in the active rubric)")
        lines.append("")
        lines.append(
            f"Count: {entry['count']}   "
            f"Docs: {entry['doc_count']}   "
            f"Avg confidence: {entry['avg_confidence']:.0%} "
            f"   Reviews: {entry['review_count']}   "
            f"Disagree: {entry['disagree_pct']:.0%}"
        )
        if flags:
            lines.append("")
            lines.append("Flags:")
            for f in flags:
                lines.append(f"  • {f}")

        if cat:
            lines.append("")
            lines.append("Definition:")
            lines.append(f"  {cat.definition or '(empty)'}")
            if cat.exemplars:
                lines.append(f"Exemplars ({len(cat.exemplars)}):")
                for ex in cat.exemplars[:3]:
                    snippet = ex if len(ex) <= 120 else ex[:120] + "…"
                    lines.append(f"  • {snippet}")
            if cat.boundary_cases:
                lines.append(f"Boundary cases ({len(cat.boundary_cases)}):")
                for bc in cat.boundary_cases[:3]:
                    snippet = bc if len(bc) <= 120 else bc[:120] + "…"
                    lines.append(f"  • {snippet}")

        self.coverage_detail.setPlainText("\n".join(lines))

    def _save_rubric(self) -> None:
        if self.ctx.rubric_manager is None:
            return

        categories = self._get_categories()
        if not categories:
            self.main_window.notification_manager.show_warning(
                "Add at least one category"
            )
            return

        try:
            rubric = self.ctx.rubric_manager.get_active_rubric()
            if rubric:
                new_rubric = self.ctx.rubric_manager.update_rubric(
                    rubric.id, categories
                )
            else:
                new_rubric = self.ctx.rubric_manager.create_rubric(
                    "Default", categories
                )

            self.version_label.setText(f"v{new_rubric.version}")
            self.main_window.right_panel.update_rubric(new_rubric)
            self.main_window.notification_manager.show_success(
                f"Rubric saved (v{new_rubric.version}). "
                "Existing labels still reflect prior versions — re-run labeling to apply."
            )

            if self.ctx.audit_manager:
                self.ctx.audit_manager.log_event(
                    "rubric_updated" if rubric else "rubric_created",
                    payload={
                        "version": new_rubric.version,
                        "categories": len(categories),
                    },
                )
            self._refresh_coverage()
        except Exception as e:
            self.main_window.notification_manager.show_error(str(e))

    def _import_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Rubric", "", "JSON (*.json)"
        )
        if not path:
            return

        try:
            with open(path) as f:
                json_str = f.read()

            if self.ctx.rubric_manager is None:
                return

            categories = self.ctx.rubric_manager.parse_categories_json(json_str)

            existing = self.ctx.rubric_manager.get_active_rubric()
            if existing:
                new_rubric = self.ctx.rubric_manager.update_rubric(
                    existing.id, categories
                )
            else:
                new_rubric = self.ctx.rubric_manager.create_rubric(
                    "Imported", categories
                )

            self._clear_cards()
            for cat in new_rubric.categories:
                self._add_category_card(cat)

            self.version_label.setText(f"v{new_rubric.version}")
            self.main_window.right_panel.update_rubric(new_rubric)
            self.main_window.notification_manager.show_success(
                f"Imported {len(categories)} categories (v{new_rubric.version})"
            )
            self._refresh_coverage()

            if self.ctx.audit_manager:
                self.ctx.audit_manager.log_event(
                    "rubric_imported",
                    payload={
                        "version": new_rubric.version,
                        "categories": len(categories),
                    },
                )
        except Exception as e:
            self.main_window.notification_manager.show_error(f"Import failed: {e}")

    def _export_json(self) -> None:
        categories = self._get_categories()
        if not categories:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Rubric", "rubric.json", "JSON (*.json)"
        )
        if not path:
            return

        data = {
            "categories": [
                {
                    "name": c.name,
                    "definition": c.definition,
                    "exemplars": c.exemplars,
                    "boundary_cases": c.boundary_cases,
                    "confidence_threshold": c.confidence_threshold,
                }
                for c in categories
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        self.main_window.notification_manager.show_success(f"Exported to {path}")
