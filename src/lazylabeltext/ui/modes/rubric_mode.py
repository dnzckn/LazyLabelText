"""Rubric mode: structured category editor with live label preview."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.core.models import Category
from lazylabeltext.ui.modes.base_mode import BaseMode
from lazylabeltext.ui.widgets.confidence_bar import ConfidenceBar
from lazylabeltext.ui.widgets.rubric_category_card import RubricCategoryCard

if TYPE_CHECKING:
    from lazylabeltext.core.app_context import AppContext
    from lazylabeltext.ui.main_window import MainWindow

logger = logging.getLogger("lazylabeltext")


class RubricModeWidget(BaseMode):
    """Smart rubric editor with live preview of label predictions."""

    def __init__(
        self,
        context: AppContext,
        main_window: MainWindow,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(context, parent)
        self.main_window = main_window
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(1500)
        self._preview_timer.timeout.connect(self._run_preview)
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

        # Right: live preview
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 8, 4)

        right_header = QLabel("Live Preview")
        right_header.setObjectName("sectionHeader")
        right_layout.addWidget(right_header)

        self.preview_info = QLabel(
            "Edit the rubric to see how chunks would be labeled."
        )
        self.preview_info.setStyleSheet("color: #888; font-size: 11px;")
        self.preview_info.setWordWrap(True)
        right_layout.addWidget(self.preview_info)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_container = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_container)
        self.preview_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.preview_layout.setSpacing(6)
        self.preview_scroll.setWidget(self.preview_container)
        right_layout.addWidget(self.preview_scroll, 1)

        splitter.addWidget(right)
        splitter.setSizes([400, 400])

        layout.addWidget(splitter, 1)

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
        """Debounced handler for any rubric edit."""
        self._preview_timer.start()

    def _run_preview(self) -> None:
        """Re-classify sample chunks against the current rubric state."""
        categories = self._get_categories()
        if not categories:
            return

        # Clear old preview
        while self.preview_layout.count():
            child = self.preview_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Get sample chunks from current document
        doc_id = self.ctx.get_ui_state("selected_document_id")
        if doc_id is None or self.ctx.chunk_manager is None:
            self.preview_info.setText(
                "Select a document and create chunks to see live preview."
            )
            return

        chunks = self.ctx.chunk_manager.get_chunks(doc_id)
        if not chunks:
            self.preview_info.setText("No chunks found. Switch to Chunk mode first.")
            return

        # Show preview for up to 5 chunks
        sample = chunks[:5]
        self.preview_info.setText(
            f"Showing {len(sample)} of {len(chunks)} chunks. "
            "Labels update as you edit the rubric."
        )

        if self.ctx.llm_provider is not None:
            # Use LLM for live preview
            for chunk in sample:
                try:
                    result = self.ctx.llm_provider.classify(chunk.text, categories)
                    self._add_preview_chunk(
                        chunk.text, result.categories, result.confidence_per_category
                    )
                except Exception:
                    self._add_preview_chunk(chunk.text, ["(error)"], {})
        else:
            # No LLM: just show chunks without labels
            for chunk in sample:
                self._add_preview_chunk(chunk.text, ["(no LLM configured)"], {})

    def _add_preview_chunk(
        self, text: str, categories: list[str], confidence: dict[str, float]
    ) -> None:
        card = QWidget()
        card.setObjectName("chunkCard")
        card.setStyleSheet("QWidget#chunkCard { border-radius: 4px; padding: 6px; }")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 6, 8, 6)
        card_layout.setSpacing(4)

        # Truncated text
        preview = text[:150] + "..." if len(text) > 150 else text
        text_label = QLabel(preview)
        text_label.setWordWrap(True)
        text_label.setStyleSheet("font-size: 11px;")
        card_layout.addWidget(text_label)

        # Labels with confidence bars
        for cat in categories:
            conf = confidence.get(cat, 0.0)
            bar = ConfidenceBar(value=conf, label=cat)
            card_layout.addWidget(bar)

        self.preview_layout.addWidget(card)

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
                f"Rubric saved (v{new_rubric.version})"
            )

            if self.ctx.audit_manager:
                self.ctx.audit_manager.log_event(
                    "rubric_updated" if rubric else "rubric_created",
                    payload={
                        "version": new_rubric.version,
                        "categories": len(categories),
                    },
                )
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
            self._clear_cards()
            for cat in categories:
                self._add_category_card(cat)
            self.main_window.notification_manager.show_success(
                f"Imported {len(categories)} categories"
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
