"""Chunk mode: smart chunk viewer with parameter tuning and document overlay."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.core.chunkers import available_strategies
from lazylabeltext.ui.modes.base_mode import BaseMode
from lazylabeltext.ui.workers.chunking_worker import ChunkingWorker

if TYPE_CHECKING:
    from lazylabeltext.core.app_context import AppContext
    from lazylabeltext.ui.main_window import MainWindow

logger = logging.getLogger("lazylabeltext")

_CHUNK_COLORS = [
    QColor(92, 143, 191, 40),
    QColor(143, 191, 92, 40),
    QColor(191, 143, 92, 40),
    QColor(143, 92, 191, 40),
    QColor(92, 191, 143, 40),
    QColor(191, 92, 143, 40),
]


class ChunkModeWidget(BaseMode):
    """Chunk viewer with overlay + card views and parameter controls."""

    def __init__(
        self,
        context: AppContext,
        main_window: MainWindow,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(context, parent)
        self.main_window = main_window
        self._current_doc_id: int | None = None
        self._current_run_id: int | None = None
        self._chunking_worker: ChunkingWorker | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: chunk views (tabbed)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 4, 4, 4)

        self.view_tabs = QTabWidget()

        # Tab 1: Document overlay
        self.overlay_view = QTextEdit()
        self.overlay_view.setReadOnly(True)
        self.overlay_view.setPlaceholderText(
            "Select a document and run chunking to see results."
        )
        self.view_tabs.addTab(self.overlay_view, "Document Overlay")

        # Tab 2: Card list
        card_scroll = QScrollArea()
        card_scroll.setWidgetResizable(True)
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.cards_layout.setSpacing(4)
        card_scroll.setWidget(self.cards_container)
        self.view_tabs.addTab(card_scroll, "Chunk Cards")

        left_layout.addWidget(self.view_tabs, 1)

        # Stats bar
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("color: #888; font-size: 11px; padding: 2px;")
        left_layout.addWidget(self.stats_label)

        splitter.addWidget(left)

        # Right: parameters panel
        right = QWidget()
        right.setMaximumWidth(300)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 8, 4)

        right_header = QLabel("Chunking Parameters")
        right_header.setObjectName("sectionHeader")
        right_layout.addWidget(right_header)

        # Strategy
        right_layout.addWidget(QLabel("Strategy:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(available_strategies())
        right_layout.addWidget(self.strategy_combo)

        # Token thresholds
        right_layout.addWidget(QLabel("Min tokens:"))
        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setRange(10, 200)
        self.min_slider.setValue(50)
        self.min_label = QLabel("50")
        self.min_slider.valueChanged.connect(lambda v: self.min_label.setText(str(v)))
        min_row = QHBoxLayout()
        min_row.addWidget(self.min_slider)
        min_row.addWidget(self.min_label)
        right_layout.addLayout(min_row)

        right_layout.addWidget(QLabel("Max tokens:"))
        self.max_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_slider.setRange(100, 2000)
        self.max_slider.setValue(800)
        self.max_label = QLabel("800")
        self.max_slider.valueChanged.connect(lambda v: self.max_label.setText(str(v)))
        max_row = QHBoxLayout()
        max_row.addWidget(self.max_slider)
        max_row.addWidget(self.max_label)
        right_layout.addLayout(max_row)

        # Heading levels
        right_layout.addWidget(QLabel("Split on heading levels:"))
        self.heading_checks: dict[int, QCheckBox] = {}
        heading_row = QHBoxLayout()
        for level in range(1, 7):
            cb = QCheckBox(f"H{level}")
            cb.setChecked(level <= 3)
            heading_row.addWidget(cb)
            self.heading_checks[level] = cb
        right_layout.addLayout(heading_row)

        # Similarity threshold (semantic / hybrid only)
        self.similarity_label_header = QLabel(
            "Similarity threshold (semantic/hybrid):"
        )
        right_layout.addWidget(self.similarity_label_header)
        self.similarity_slider = QSlider(Qt.Orientation.Horizontal)
        self.similarity_slider.setRange(10, 90)  # 0.10 - 0.90
        self.similarity_slider.setValue(50)
        self.similarity_label = QLabel("0.50")
        self.similarity_slider.valueChanged.connect(
            lambda v: self.similarity_label.setText(f"{v / 100:.2f}")
        )
        sim_row = QHBoxLayout()
        sim_row.addWidget(self.similarity_slider)
        sim_row.addWidget(self.similarity_label)
        right_layout.addLayout(sim_row)
        self._update_similarity_visibility(self.strategy_combo.currentText())
        self.strategy_combo.currentTextChanged.connect(
            self._update_similarity_visibility
        )

        # Chunk button
        self.chunk_btn = QPushButton("Run Chunking")
        self.chunk_btn.setObjectName("accentButton")
        self.chunk_btn.clicked.connect(self._run_chunking)
        right_layout.addWidget(self.chunk_btn)

        right_layout.addStretch()

        splitter.addWidget(right)
        splitter.setSizes([600, 250])

        layout.addWidget(splitter, 1)

    def on_document_selected(self, doc_id: int) -> None:
        self._current_doc_id = doc_id
        self._show_existing_chunks()

    def activate(self) -> None:
        doc_id = self.ctx.get_ui_state("selected_document_id")
        if doc_id:
            self._current_doc_id = doc_id
            self._show_existing_chunks()

    def _get_params(self) -> dict:
        levels = [lv for lv, cb in self.heading_checks.items() if cb.isChecked()]
        return {
            "min_tokens": self.min_slider.value(),
            "max_tokens": self.max_slider.value(),
            "heading_split_levels": levels,
            "similarity_threshold": self.similarity_slider.value() / 100.0,
        }

    def _update_similarity_visibility(self, strategy: str) -> None:
        needs_threshold = strategy in ("semantic", "hybrid")
        self.similarity_label_header.setVisible(needs_threshold)
        self.similarity_slider.setVisible(needs_threshold)
        self.similarity_label.setVisible(needs_threshold)

    def _run_chunking(self) -> None:
        if self._current_doc_id is None or self.ctx.chunk_manager is None:
            return
        if self._chunking_worker is not None and self._chunking_worker.isRunning():
            return

        strategy = self.strategy_combo.currentText()
        params = self._get_params()

        self.chunk_btn.setEnabled(False)
        self.stats_label.setText(f"Chunking with '{strategy}'...")

        worker = ChunkingWorker(
            self.ctx.chunk_manager,
            self._current_doc_id,
            strategy,
            params,
            parent=self,
        )
        worker.progress.connect(self._on_chunking_progress)
        worker.finished_with_run.connect(self._on_chunking_finished)
        worker.error.connect(self._on_chunking_error)
        self._chunking_worker = worker
        worker.start()

    def _on_chunking_progress(self, current: int, total: int) -> None:
        self.stats_label.setText(f"Chunking... ({current}/{total} windows)")

    def _on_chunking_finished(self, run_id: int) -> None:
        self._current_run_id = run_id if run_id else None
        self.chunk_btn.setEnabled(True)
        self._chunking_worker = None
        if run_id:
            self._show_chunks_for_run(run_id)
            self.main_window._update_stats()
            try:
                runs = self.ctx.chunk_manager.get_chunking_runs(
                    self._current_doc_id or 0
                )
                latest = runs[-1] if runs else None
                n = latest.n_chunks if latest else 0
            except Exception:
                n = 0
            self.main_window.notification_manager.show_success(
                f"Created {n} chunks"
            )

    def _on_chunking_error(self, msg: str) -> None:
        self.chunk_btn.setEnabled(True)
        self._chunking_worker = None
        self.stats_label.setText("Chunking failed.")
        self.main_window.notification_manager.show_error(f"Chunking failed: {msg}")

    def _show_existing_chunks(self) -> None:
        """Show chunks for the currently selected document."""
        if self._current_doc_id is None or self.ctx.chunk_manager is None:
            return

        runs = self.ctx.chunk_manager.get_chunking_runs(self._current_doc_id)
        if runs:
            latest = runs[-1]
            self._current_run_id = latest.id
            self._show_chunks_for_run(latest.id)
        else:
            self._clear_views()
            self.stats_label.setText("No chunks yet. Click 'Run Chunking'.")

    def _show_chunks_for_run(self, run_id: int | None) -> None:
        if run_id is None or self.ctx.chunk_manager is None:
            return

        chunks = self.ctx.chunk_manager.get_chunks(self._current_doc_id or 0, run_id)
        stats = self.ctx.chunk_manager.get_chunk_statistics(run_id)

        # Update stats
        self.stats_label.setText(
            f"{stats['count']} chunks | "
            f"Tokens: {stats.get('min', 0)}-{stats.get('max', 0)} "
            f"(mean: {stats.get('mean', 0)}, median: {stats.get('median', 0)})"
        )

        # Update overlay view
        self._render_overlay(chunks)

        # Update card view
        self._render_cards(chunks)

        # Update left panel chunk count
        if self._current_doc_id:
            self.main_window.left_panel.update_chunk_count(
                self._current_doc_id, len(chunks)
            )

    def _render_overlay(self, chunks) -> None:
        """Render document text with chunk boundaries highlighted."""
        if (
            not chunks
            or self.ctx.document_manager is None
            or self._current_doc_id is None
        ):
            return

        doc = self.ctx.document_manager.get_document(self._current_doc_id)
        if doc is None:
            return

        self.overlay_view.clear()
        self.overlay_view.setPlainText(doc.full_text)

        cursor = self.overlay_view.textCursor()
        for i, chunk in enumerate(chunks):
            color = _CHUNK_COLORS[i % len(_CHUNK_COLORS)]
            fmt = QTextCharFormat()
            fmt.setBackground(color)

            cursor.setPosition(chunk.char_start)
            cursor.setPosition(
                min(chunk.char_end, len(doc.full_text)),
                QTextCursor.MoveMode.KeepAnchor,
            )
            cursor.mergeCharFormat(fmt)

    def _render_cards(self, chunks) -> None:
        """Render chunk cards."""
        # Clear existing
        while self.cards_layout.count():
            child = self.cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for i, chunk in enumerate(chunks):
            card = self._make_chunk_card(chunk, i)
            self.cards_layout.addWidget(card)

    def _make_chunk_card(self, chunk, index: int) -> QFrame:
        card = QFrame()
        card.setObjectName("chunkCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Header: index + token count
        header = QHBoxLayout()
        idx_label = QLabel(f"#{index + 1}")
        idx_label.setStyleSheet("font-weight: bold;")
        header.addWidget(idx_label)

        tokens = chunk.token_count
        token_color = "#51cf66" if 50 <= tokens <= 800 else "#ff6b6b"
        token_label = QLabel(f"{tokens} tokens")
        token_label.setStyleSheet(f"color: {token_color}; font-size: 11px;")
        header.addWidget(token_label)

        header.addStretch()

        # Section path
        if chunk.section_path:
            path_label = QLabel(" > ".join(chunk.section_path))
            path_label.setStyleSheet("color: #888; font-size: 10px;")
            header.addWidget(path_label)
        layout.addLayout(header)

        # Text preview
        text = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(text_label)

        # Action buttons
        btn_row = QHBoxLayout()
        if index > 0:
            merge_btn = QPushButton("Merge with prev")
            merge_btn.setFixedHeight(24)
            merge_btn.clicked.connect(
                lambda _, c=chunk, idx=index: self._merge_with_prev(c, idx)
            )
            btn_row.addWidget(merge_btn)

        split_btn = QPushButton("Split")
        split_btn.setFixedHeight(24)
        split_btn.clicked.connect(lambda _, c=chunk: self._split_chunk(c))
        btn_row.addWidget(split_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        return card

    def _merge_with_prev(self, chunk, index: int) -> None:
        if self.ctx.chunk_manager is None or self._current_run_id is None:
            return

        chunks = self.ctx.chunk_manager.get_chunks(
            self._current_doc_id or 0, self._current_run_id
        )
        if index <= 0 or index >= len(chunks):
            return

        prev = chunks[index - 1]
        try:
            self.ctx.chunk_manager.merge_chunks(prev.id, chunk.id)
            self._show_chunks_for_run(self._current_run_id)
            self.main_window.notification_manager.show("Chunks merged")
        except Exception as e:
            self.main_window.notification_manager.show_error(str(e))

    def _split_chunk(self, chunk) -> None:
        if self.ctx.chunk_manager is None:
            return

        # Split at midpoint
        mid = len(chunk.text) // 2
        # Try to find a paragraph boundary near midpoint
        for offset in range(0, min(100, mid)):
            if mid + offset < len(chunk.text) and chunk.text[mid + offset] == "\n":
                mid = mid + offset + 1
                break
            if mid - offset > 0 and chunk.text[mid - offset] == "\n":
                mid = mid - offset + 1
                break

        try:
            self.ctx.chunk_manager.split_chunk(chunk.id, mid)
            self._show_chunks_for_run(self._current_run_id)
            self.main_window.notification_manager.show("Chunk split")
        except Exception as e:
            self.main_window.notification_manager.show_error(str(e))

    def _clear_views(self) -> None:
        self.overlay_view.clear()
        while self.cards_layout.count():
            child = self.cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
