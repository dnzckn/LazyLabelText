"""Main application window for LazyLabelText."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from lazylabeltext.config import HotkeyManager, Paths, Settings
from lazylabeltext.core.app_context import AppContext
from lazylabeltext.core.audit_manager import AuditManager
from lazylabeltext.core.chunk_manager import ChunkManager
from lazylabeltext.core.database import Database
from lazylabeltext.core.document_manager import DocumentManager
from lazylabeltext.core.label_manager import LabelManager
from lazylabeltext.core.parallel_orchestrator import ParallelOrchestrator
from lazylabeltext.core.providers import create_embedding_provider, create_llm_provider
from lazylabeltext.core.rubric_manager import RubricManager
from lazylabeltext.ui.center_panel import CenterPanel
from lazylabeltext.ui.left_panel import LeftPanel
from lazylabeltext.ui.managers.document_navigation_manager import (
    DocumentNavigationManager,
)
from lazylabeltext.ui.managers.keyboard_event_manager import KeyboardEventManager
from lazylabeltext.ui.managers.mode_manager import MODES, ModeManager
from lazylabeltext.ui.managers.notification_manager import NotificationManager
from lazylabeltext.ui.right_panel import RightPanel
from lazylabeltext.ui.theme import apply_theme
from lazylabeltext.ui.widgets.provider_settings_dialog import ProviderSettingsDialog
from lazylabeltext.ui.widgets.status_bar import StatusBar
from lazylabeltext.ui.workers.conversion_worker import ConversionWorker

logger = logging.getLogger("lazylabeltext")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()

        # 1. Configuration
        self.paths = Paths()
        self.settings = Settings.load_from_file(str(self.paths.settings_file))
        self.hotkey_manager = HotkeyManager(str(self.paths.config_dir))

        # 2. Core managers (database set when project is opened)
        self.database: Database | None = None
        self.document_manager: DocumentManager | None = None
        self.rubric_manager: RubricManager | None = None
        self.chunk_manager: ChunkManager | None = None
        self.label_manager: LabelManager | None = None
        self.audit_manager: AuditManager | None = None
        self.parallel_orchestrator: ParallelOrchestrator | None = None

        # 3. Providers
        self.llm_provider = None
        self.embedding_provider = None
        self._init_providers()

        # 4. App context
        self.app_context = AppContext(
            paths=self.paths,
            settings=self.settings,
            hotkey_manager=self.hotkey_manager,
        )

        # 5. UI setup
        self._mode_buttons: dict[str, QPushButton] = {}
        self._setup_ui()

        # 6. UI managers
        self.mode_manager = ModeManager(self)
        self.notification_manager = NotificationManager(self)
        self.doc_nav_manager = DocumentNavigationManager(self)
        self.keyboard_manager = KeyboardEventManager(self)

        # 7. Connect signals
        self._connect_signals()
        self._setup_shortcuts()

        # 8. Window geometry + icon
        from lazylabeltext import __version__

        self._title_base = f"LazyLabelText by Deniz N. Cakan (version {__version__})"
        self.setWindowTitle(self._title_base)
        self.resize(self.settings.window_width, self.settings.window_height)

        if self.paths.logo_path.exists():
            from PyQt6.QtGui import QIcon

            self.setWindowIcon(QIcon(str(self.paths.logo_path)))

        # 9. Update status bar
        self._update_provider_status()

    def _init_providers(self) -> None:
        """Initialize LLM and embedding providers from settings."""
        provider_name = self.settings.llm_provider
        use_env = getattr(self.settings, "llm_use_env_credentials", True)

        llm_kwargs: dict = {"model": self.settings.llm_model}
        if not use_env and self.settings.llm_api_key:
            llm_kwargs["api_key"] = self.settings.llm_api_key
        if provider_name == "ollama" and self.settings.llm_base_url:
            llm_kwargs["base_url"] = self.settings.llm_base_url
        if provider_name == "azure":
            llm_kwargs["api_version"] = getattr(
                self.settings, "llm_azure_api_version", "2024-08-01-preview"
            )
            endpoint = getattr(self.settings, "llm_azure_endpoint", "")
            if endpoint:
                llm_kwargs["azure_endpoint"] = endpoint
            llm_kwargs["verify_ssl"] = getattr(
                self.settings, "llm_azure_verify_ssl", True
            )
            llm_kwargs["http2"] = getattr(self.settings, "llm_azure_http2", True)
            llm_kwargs["use_env_credentials"] = use_env

        self.llm_provider = create_llm_provider(provider_name, **llm_kwargs)

        # Embedding provider — Azure shares the LLM Azure config.
        emb_kwargs: dict = {"model_name": self.settings.embedding_model}
        emb_provider = self.settings.embedding_provider
        if emb_provider in ("openai", "openai-embeddings"):
            emb_kwargs["api_key"] = (
                getattr(self.settings, "embedding_api_key", "")
                or self.settings.llm_api_key  # share with LLM key when both are OpenAI
            )
        elif emb_provider in ("azure", "azure-embeddings"):
            emb_kwargs["api_version"] = getattr(
                self.settings, "llm_azure_api_version", "2024-08-01-preview"
            )
            endpoint = getattr(self.settings, "llm_azure_endpoint", "")
            if endpoint:
                emb_kwargs["azure_endpoint"] = endpoint
            emb_kwargs["verify_ssl"] = getattr(
                self.settings, "llm_azure_verify_ssl", True
            )
            emb_kwargs["http2"] = getattr(self.settings, "llm_azure_http2", True)
            emb_kwargs["use_env_credentials"] = use_env
            if not use_env:
                emb_kwargs["api_key"] = (
                    getattr(self.settings, "embedding_api_key", "")
                    or self.settings.llm_api_key
                )
        self.embedding_provider = create_embedding_provider(
            emb_provider, **emb_kwargs
        )

    def _setup_ui(self) -> None:
        """Build the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Toolbar ---
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(4)

        self.mode_button_group = QButtonGroup(self)
        self.mode_button_group.setExclusive(True)

        for mode in MODES:
            btn = QPushButton(mode.title())
            btn.setObjectName("modeButton")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, m=mode: self._on_mode_button_clicked(m))
            toolbar_layout.addWidget(btn)
            self.mode_button_group.addButton(btn)
            self._mode_buttons[mode] = btn

        self._mode_buttons["convert"].setChecked(True)

        toolbar_layout.addStretch()

        # Settings gear button
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._open_settings)
        toolbar_layout.addWidget(settings_btn)

        main_layout.addWidget(toolbar)

        # --- Three-panel splitter ---
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.left_panel = LeftPanel()
        self.main_splitter.addWidget(self.left_panel)

        self.center_panel = CenterPanel()
        self.main_splitter.addWidget(self.center_panel)

        self.right_panel = RightPanel()
        self.main_splitter.addWidget(self.right_panel)

        self.main_splitter.setSizes(
            [
                self.settings.left_panel_width,
                self.settings.window_width
                - self.settings.left_panel_width
                - self.settings.right_panel_width,
                self.settings.right_panel_width,
            ]
        )
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)

        main_layout.addWidget(self.main_splitter, 1)

        # --- Status bar ---
        self.status_bar = StatusBar(dark_mode=self.settings.dark_mode)
        self.setStatusBar(self.status_bar)

    def _connect_signals(self) -> None:
        """Wire up signal-slot connections."""
        self.left_panel.open_folder_requested.connect(self._open_folder_dialog)
        self.left_panel.open_docmap_requested.connect(self._open_docmap_dialog)
        self.left_panel.document_selected.connect(self._on_document_selected)
        self.left_panel.reset_project_requested.connect(self._reset_project)
        self.left_panel.document_clear_requested.connect(
            self._clear_document_data
        )
        self.left_panel.document_delete_requested.connect(
            self._delete_document_from_project
        )
        self.right_panel.rubric_edit_requested.connect(
            lambda: self.mode_manager.set_mode("rubric")
        )
        self.status_bar.theme_toggled.connect(self._on_theme_toggled)

    def _setup_shortcuts(self) -> None:
        """Create QShortcuts from hotkey manager."""
        action_callbacks = {
            "next_document": self.doc_nav_manager.select_next,
            "previous_document": self.doc_nav_manager.select_previous,
            "convert_mode": lambda: self.mode_manager.set_mode("convert"),
            "rubric_mode": lambda: self.mode_manager.set_mode("rubric"),
            "chunk_mode": lambda: self.mode_manager.set_mode("chunk"),
            "label_mode": lambda: self.mode_manager.set_mode("label"),
            "results_mode": lambda: self.mode_manager.set_mode("results"),
            "parallel_mode": lambda: self.mode_manager.set_mode("parallel"),
            "export_mode": lambda: self.mode_manager.set_mode("export"),
            "save_project": self._save_project,
            "open_folder": self._open_folder_dialog,
            "settings": self._open_settings,
            "toggle_theme": self._toggle_theme,
            "accept_label": self.keyboard_manager.handle_space,
        }

        for action_name, callback in action_callbacks.items():
            primary, secondary = self.hotkey_manager.get_key_for_action(action_name)
            if primary:
                shortcut = QShortcut(QKeySequence(primary), self)
                shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
                shortcut.activated.connect(callback)
            if secondary:
                shortcut = QShortcut(QKeySequence(secondary), self)
                shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
                shortcut.activated.connect(callback)

    # --- Project management ---

    def open_project(self, folder_path: str) -> None:
        """Open or create a folder-backed project."""
        db_path = Path(folder_path) / "project.db"
        existing_count = self._init_project_at(
            db_path=db_path, title_label=Path(folder_path).name,
        )
        # Persist as a folder-style project for auto-reopen.
        self.settings.last_project_kind = "folder"
        self.settings.last_project_path = folder_path
        self.settings.last_project_output = folder_path
        self.settings.last_docmap_path = ""
        self.settings.save_to_file(str(self.paths.settings_file))

        if self.document_manager is None:
            return
        try:
            paths = self.document_manager.find_supported_files(folder_path)
        except FileNotFoundError as e:
            self.notification_manager.show_error(str(e))
            return
        self._start_conversion_for_paths(paths, existing_count)

    def open_project_from_docmap(
        self, docmap_path: str, output_dir: str
    ) -> None:
        """Open or create a doc-map-backed project.

        Source documents stay where they live; only project.db lives in
        ``output_dir``. The doc-map is re-resolved against the filesystem
        on every open, so newly-added files matching its patterns are
        picked up automatically.
        """
        from lazylabeltext.core.doc_map import DocMapError

        out = Path(output_dir)
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.notification_manager.show_error(
                f"Could not create output dir {out}: {e}"
            )
            return
        db_path = out / "project.db"

        existing_count = self._init_project_at(
            db_path=db_path, title_label=Path(docmap_path).name,
        )
        self.settings.last_project_kind = "docmap"
        self.settings.last_docmap_path = str(Path(docmap_path).resolve())
        self.settings.last_project_output = str(out.resolve())
        self.settings.last_project_path = ""  # only meaningful in folder mode
        self.settings.save_to_file(str(self.paths.settings_file))

        if self.document_manager is None:
            return
        try:
            paths, result = (
                self.document_manager.find_supported_files_from_docmap(
                    docmap_path
                )
            )
        except DocMapError as e:
            self.notification_manager.show_error(f"Doc map error: {e}")
            return

        # Surface any include patterns that matched zero files so the user
        # knows their map is partly stale (renamed dir, typo, etc.).
        for entry in result.unmatched_patterns:
            logger.warning(
                "Doc map %s:%d pattern %r matched no files",
                docmap_path, entry.line_no, entry.pattern,
            )
        if result.unmatched_patterns:
            self.notification_manager.show_warning(
                f"{len(result.unmatched_patterns)} doc-map pattern(s) "
                "matched no files (see log)"
            )

        self._start_conversion_for_paths(paths, existing_count)

    def _init_project_at(self, db_path: Path, title_label: str) -> int:
        """Open the database, wire managers + context, return existing-doc count.

        Shared between folder-open and docmap-open flows. Returns the count
        of documents already persisted in the DB so the caller can craft a
        useful "Loaded N docs" toast after conversion completes.
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.database = Database(str(db_path))

        # Initialize managers
        self.document_manager = DocumentManager(self.database, self.settings)
        self.rubric_manager = RubricManager(self.database)
        self.chunk_manager = ChunkManager(
            self.database, self.embedding_provider, self.llm_provider
        )
        self.label_manager = LabelManager(
            self.database, self.llm_provider, self.embedding_provider
        )
        self.audit_manager = AuditManager(self.database)
        self.parallel_orchestrator = ParallelOrchestrator(
            self.database,
            self.document_manager,
            self.chunk_manager,
            self.label_manager,
            self.settings,
        )

        # Update app context
        self.app_context.database = self.database
        self.app_context.document_manager = self.document_manager
        self.app_context.rubric_manager = self.rubric_manager
        self.app_context.chunk_manager = self.chunk_manager
        self.app_context.label_manager = self.label_manager
        self.app_context.audit_manager = self.audit_manager
        self.app_context.parallel_orchestrator = self.parallel_orchestrator
        self.app_context.llm_provider = self.llm_provider
        self.app_context.embedding_provider = self.embedding_provider

        # Show docs that already exist in the DB straight away so the panel
        # isn't empty while new files are converting in the background.
        existing_docs = self.document_manager.get_all_documents()
        self.left_panel.populate(existing_docs)

        # Load rubric
        rubric = self.rubric_manager.get_active_rubric()
        self.right_panel.update_rubric(rubric)

        # Initialize mode widgets with context
        self._init_mode_widgets()

        # Step 1 of the workflow is converting — open there by default.
        self.mode_manager.set_mode("convert")
        self._update_stats()

        self.setWindowTitle(f"{self._title_base} — {title_label}")
        return len(existing_docs)

    def _start_conversion_for_paths(
        self, all_paths: list[str], existing_count: int
    ) -> None:
        """Convert any new paths in a worker. Source can be folder or docmap."""
        if self.document_manager is None:
            return

        # Per-content dedupe: every conversion call inside load_single hashes
        # its source and looks the doc up by SHA256. Since identity is the
        # content hash (not the filename), main_window doesn't need to
        # filter out anything here — moves are silently merged, and two
        # genuinely-different files that share a basename are loaded as
        # separate rows. The only thing we still need to guard against is
        # *the same physical path* listed twice in this run's paths.
        seen_paths: set[str] = set()
        new_paths: list[str] = []
        for p in all_paths:
            if p in seen_paths:
                continue
            seen_paths.add(p)
            new_paths.append(p)

        if not new_paths:
            self.notification_manager.show_success(
                f"Opened project: {existing_count} documents"
            )
            return

        progress = QProgressDialog(
            "Converting documents…", "Cancel", 0, len(new_paths), self
        )
        progress.setWindowTitle("Converting")
        progress.setMinimumDuration(0)
        progress.setAutoClose(True)
        progress.setAutoReset(False)
        progress.setValue(0)

        max_workers = max(
            1, getattr(self.settings, "conversion_workers", 1)
        )
        worker = ConversionWorker(
            self.document_manager, new_paths,
            max_workers=max_workers, parent=self,
        )

        def _on_progress(current: int, total: int) -> None:
            progress.setMaximum(total)
            progress.setValue(current)

        def _on_status(msg: str) -> None:
            if msg:
                progress.setLabelText(msg)
                self.notification_manager.show(msg, duration=0)

        def _on_doc(doc_id: int) -> None:
            if self.document_manager is None:
                return
            doc = self.document_manager.get_document(doc_id)
            if doc is None:
                return
            self.left_panel.populate(self.document_manager.get_all_documents())

        def _on_failed(_doc_id: int, err: str) -> None:
            # Surface failed conversions in the left panel so the user can see
            # the file was attempted (and why it failed), instead of looking
            # silently ignored. record_failed already inserted a 'failed' row;
            # populate the panel from the DB to include it.
            logger.warning("Conversion failed: %s", err)
            if self.document_manager is None:
                return
            self.left_panel.populate(self.document_manager.get_all_documents())

        def _on_finished() -> None:
            progress.setValue(progress.maximum())
            progress.close()
            if self.document_manager is None:
                return
            # Final sync — guarantees both succeeded and failed rows are
            # represented in the panel even if a signal got dropped.
            all_docs = self.document_manager.get_all_documents()
            self.left_panel.populate(all_docs)
            self._update_stats()
            failed = sum(1 for d in all_docs if d.status == "failed")
            ok = len(all_docs) - failed
            if failed:
                self.notification_manager.show_warning(
                    f"Opened project: {ok} ok, {failed} failed (see panel for details)"
                )
            else:
                self.notification_manager.show_success(
                    f"Opened project: {ok} documents"
                )
            self._conversion_worker = None

        def _on_error(err: str) -> None:
            progress.close()
            self.notification_manager.show_error(f"Conversion error: {err}")
            self._conversion_worker = None

        worker.progress.connect(_on_progress)
        worker.status_changed.connect(_on_status)
        worker.document_converted.connect(_on_doc)
        worker.document_failed.connect(_on_failed)
        worker.finished.connect(_on_finished)
        worker.error.connect(_on_error)
        progress.canceled.connect(worker.stop)

        # Hold a reference so Qt doesn't garbage-collect mid-run.
        self._conversion_worker = worker
        worker.start()

    def _init_mode_widgets(self) -> None:
        """Initialize mode-specific widgets with app context."""
        from lazylabeltext.ui.modes.chunk_mode import ChunkModeWidget
        from lazylabeltext.ui.modes.convert_mode import ConvertModeWidget
        from lazylabeltext.ui.modes.export_mode import ExportModeWidget
        from lazylabeltext.ui.modes.label_mode import LabelModeWidget
        from lazylabeltext.ui.modes.parallel_mode import ParallelModeWidget
        from lazylabeltext.ui.modes.results_mode import ResultsModeWidget
        from lazylabeltext.ui.modes.rubric_mode import RubricModeWidget

        self.center_panel.set_mode_widget(
            "convert", ConvertModeWidget(self.app_context)
        )
        self.center_panel.set_mode_widget(
            "rubric", RubricModeWidget(self.app_context, self)
        )
        self.center_panel.set_mode_widget(
            "chunk", ChunkModeWidget(self.app_context, self)
        )
        self.center_panel.set_mode_widget(
            "label", LabelModeWidget(self.app_context, self)
        )
        self.center_panel.set_mode_widget(
            "results", ResultsModeWidget(self.app_context)
        )
        self.center_panel.set_mode_widget(
            "parallel", ParallelModeWidget(self.app_context, self)
        )
        self.center_panel.set_mode_widget("export", ExportModeWidget(self.app_context))

    def _update_stats(self) -> None:
        """Update status bar + per-document stats in the left panel."""
        if self.database is None:
            self.status_bar.set_project_stats()
            return

        docs = (
            len(self.document_manager.get_all_documents())
            if self.document_manager
            else 0
        )
        chunks = 0
        labels = 0

        status = self.database.get_document_label_status()
        for s in status:
            chunks += s["chunk_count"]
            labels += s["label_count"]

        self.status_bar.set_project_stats(docs, chunks, labels)
        self.left_panel.update_document_stats(status)

    def _clear_document_data(self, doc_id: int) -> None:
        """Wipe chunks + labels + reviews + chunking_runs for one document."""
        if self.database is None:
            return
        doc = self.document_manager.get_document(doc_id) if self.document_manager else None
        name = doc.filename if doc else "this document"

        confirm = QMessageBox.question(
            self,
            "Clear data for this document?",
            f"This will wipe every chunk, label, review, and chunking run "
            f"for '{name}'. The document itself stays in the project; the "
            f"source file on disk is untouched. This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            self.database.delete_all_chunking_for_document(doc_id)
            if self.audit_manager:
                self.audit_manager.log_event(
                    "document_data_cleared", payload={"document_id": doc_id}
                )
            self.notification_manager.show_success(f"Cleared data for {name}")
        except Exception as e:
            self.notification_manager.show_error(f"Clear failed: {e}")
            return

        self._update_stats()
        # Refresh whichever mode is active.
        current = self.center_panel.current_mode
        widget = self.center_panel.get_mode_widget(current)
        if widget and hasattr(widget, "activate"):
            widget.activate()

    def _delete_document_from_project(self, doc_id: int) -> None:
        """Remove the document row + cascade-delete every related record."""
        if self.database is None:
            return
        doc = self.document_manager.get_document(doc_id) if self.document_manager else None
        name = doc.filename if doc else "this document"

        confirm = QMessageBox.question(
            self,
            "Remove document from project?",
            f"This will remove '{name}' from the project, including its "
            f"chunks, labels, reviews, and chunking runs. The source file "
            f"on disk is NOT deleted — you can re-add it via Open Folder. "
            f"This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            self.database.delete_document(doc_id)
            if self.audit_manager:
                self.audit_manager.log_event(
                    "document_removed", payload={"document_id": doc_id}
                )
            self.notification_manager.show_success(
                f"Removed {name} from project"
            )
        except Exception as e:
            self.notification_manager.show_error(f"Remove failed: {e}")
            return

        # Repopulate the tree from the DB.
        if self.document_manager:
            docs = self.document_manager.get_all_documents()
            self.left_panel.populate(docs)
        self._update_stats()
        current = self.center_panel.current_mode
        widget = self.center_panel.get_mode_widget(current)
        if widget and hasattr(widget, "activate"):
            widget.activate()

    def _update_provider_status(self) -> None:
        """Update provider indicator in status bar."""
        from lazylabeltext.ai_availability import EMBEDDING_AVAILABLE, LLM_AVAILABLE

        parts = []
        if self.llm_provider:
            parts.append(f"LLM: {self.settings.llm_model}")
        elif LLM_AVAILABLE:
            parts.append("LLM: No API key")
        else:
            parts.append("LLM: Not installed")

        if self.embedding_provider:
            parts.append("Emb: Ready")
        elif EMBEDDING_AVAILABLE:
            parts.append("Emb: Available")
        else:
            parts.append("Emb: Not installed")

        self.status_bar.set_provider_status(" | ".join(parts))

    # --- Slots ---

    def _on_mode_button_clicked(self, mode: str) -> None:
        self.mode_manager.set_mode(mode)

    def _on_document_selected(self, doc_id: int) -> None:
        """Handle document selection from left panel."""
        self.app_context.set_ui_state("selected_document_id", doc_id)
        # Notify active mode widget
        mode = self.mode_manager.current_mode
        widget = self.center_panel.get_mode_widget(mode)
        if widget and hasattr(widget, "on_document_selected"):
            widget.on_document_selected(doc_id)

    def _reset_project(self) -> None:
        """Wipe project.db and reopen the project at the same folder."""
        if self.database is None:
            self.notification_manager.show_warning("No project open.")
            return

        from pathlib import Path

        db_path = Path(self.database.db_path)
        folder = db_path.parent

        confirm = QMessageBox.question(
            self,
            "Reset project?",
            f"This will delete {db_path.name} and erase every chunk, label, "
            f"review, and audit event for this project.\n\n"
            f"The source documents in {folder} are NOT touched.\n\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            self.database.close()
        except Exception:
            pass

        try:
            db_path.unlink(missing_ok=True)
        except Exception as e:
            self.notification_manager.show_error(f"Could not delete DB: {e}")
            return

        self.open_project(str(folder))
        self.notification_manager.show_success("Project reset.")

    def _open_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Document Folder")
        if folder:
            self.open_project(folder)

    def _open_docmap_dialog(self) -> None:
        """Pick a doc-map file + an output dir, then open the project.

        Two-step picker so the user explicitly chooses where the project
        database lives — doc-map projects don't have a "corpus folder" we
        could implicitly drop project.db into.
        """
        docmap_path, _ = QFileDialog.getOpenFileName(
            self, "Open Doc Map",
            "",
            "Doc map (*.docmap *.txt);;All files (*)",
        )
        if not docmap_path:
            return
        # Default the output-dir picker to the docmap's parent dir.
        default_out = str(Path(docmap_path).resolve().parent)
        output_dir = QFileDialog.getExistingDirectory(
            self, "Choose project output directory (project.db lives here)",
            default_out,
        )
        if not output_dir:
            return
        self.open_project_from_docmap(docmap_path, output_dir)

    def _open_settings(self) -> None:
        dialog = ProviderSettingsDialog(self.settings, self)
        if dialog.exec():
            # Reload .env first so freshly-enabled keys land in os.environ
            # before _init_providers reads from there.
            try:
                from lazylabeltext.utils.dotenv_loader import load_dotenv_file

                if self.settings.dotenv_enabled:
                    path = (
                        self.settings.dotenv_path
                        or str(self.paths.config_dir / ".env")
                    )
                    n = load_dotenv_file(path)
                    if n:
                        self.notification_manager.show_success(
                            f"Loaded {n} variable(s) from .env"
                        )
            except Exception:
                pass

            self._init_providers()
            if self.label_manager:
                self.label_manager.set_llm_provider(self.llm_provider)
                self.label_manager.set_embedding_provider(self.embedding_provider)
            if self.chunk_manager:
                self.chunk_manager.set_embedding_provider(self.embedding_provider)
                self.chunk_manager.set_llm_provider(self.llm_provider)
            self._update_provider_status()
            self.settings.save_to_file(str(self.paths.settings_file))
            self.notification_manager.show_success("Settings saved")

    def _on_theme_toggled(self, dark: bool) -> None:
        self.settings.dark_mode = dark
        apply_theme("dark" if dark else "light")
        self.status_bar.update_theme(dark)

    def _toggle_theme(self) -> None:
        self._on_theme_toggled(not self.settings.dark_mode)

    def _save_project(self) -> None:
        # A project is "loaded" when either the folder path or the docmap
        # path has been recorded; both flows write to settings on open.
        if (
            self.settings.last_project_path
            or self.settings.last_docmap_path
        ):
            self.settings.save_to_file(str(self.paths.settings_file))
            self.notification_manager.show_success("Project saved")

    # --- Lifecycle ---

    def closeEvent(self, event) -> None:
        """Save state and clean up on close."""
        size = self.size()
        self.settings.window_width = size.width()
        self.settings.window_height = size.height()

        splitter_sizes = self.main_splitter.sizes()
        if len(splitter_sizes) >= 3:
            self.settings.left_panel_width = splitter_sizes[0]
            self.settings.right_panel_width = splitter_sizes[2]

        self.settings.save_to_file(str(self.paths.settings_file))
        self.hotkey_manager.save_hotkeys()

        if self.database:
            self.database.close()

        super().closeEvent(event)
