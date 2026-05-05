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
from lazylabeltext.core.propagation_manager import PropagationManager
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
        self.propagation_manager: PropagationManager | None = None

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

        # 8. Window geometry
        self.setWindowTitle("LazyLabelText")
        self.resize(self.settings.window_width, self.settings.window_height)

        # 9. Update status bar
        self._update_provider_status()

    def _init_providers(self) -> None:
        """Initialize LLM and embedding providers from settings."""
        if self.settings.llm_api_key or self.settings.llm_provider == "ollama":
            self.llm_provider = create_llm_provider(
                self.settings.llm_provider,
                api_key=self.settings.llm_api_key,
                model=self.settings.llm_model,
            )
        else:
            # Try from environment variable
            self.llm_provider = create_llm_provider(
                self.settings.llm_provider,
                model=self.settings.llm_model,
            )

        self.embedding_provider = create_embedding_provider(
            self.settings.embedding_provider,
            model_name=self.settings.embedding_model,
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
        self.left_panel.document_selected.connect(self._on_document_selected)
        self.left_panel.reset_project_requested.connect(self._reset_project)
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
            "propagation_mode": lambda: self.mode_manager.set_mode("propagation"),
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
        """Open or create a project in the given folder."""
        db_path = Path(folder_path) / "project.db"
        self.database = Database(str(db_path))

        # Initialize managers
        self.document_manager = DocumentManager(self.database)
        self.rubric_manager = RubricManager(self.database)
        self.chunk_manager = ChunkManager(
            self.database, self.embedding_provider, self.llm_provider
        )
        self.label_manager = LabelManager(
            self.database, self.llm_provider, self.embedding_provider
        )
        self.audit_manager = AuditManager(self.database)
        self.propagation_manager = PropagationManager(
            self.database,
            self.document_manager,
            self.chunk_manager,
            self.label_manager,
        )

        # Update app context
        self.app_context.database = self.database
        self.app_context.document_manager = self.document_manager
        self.app_context.rubric_manager = self.rubric_manager
        self.app_context.chunk_manager = self.chunk_manager
        self.app_context.label_manager = self.label_manager
        self.app_context.audit_manager = self.audit_manager
        self.app_context.llm_provider = self.llm_provider
        self.app_context.embedding_provider = self.embedding_provider

        # Load documents
        docs = self.document_manager.load_folder(folder_path)
        self.left_panel.populate(docs)

        # Load rubric
        rubric = self.rubric_manager.get_active_rubric()
        self.right_panel.update_rubric(rubric)

        # Initialize mode widgets with context
        self._init_mode_widgets()

        # Step 1 of the workflow is chunking — open there by default.
        self.mode_manager.set_mode("chunk")

        # Update stats
        self._update_stats()

        # Save last project
        self.settings.last_project_path = folder_path
        self.settings.save_to_file(str(self.paths.settings_file))

        self.setWindowTitle(f"LazyLabelText - {Path(folder_path).name}")
        self.notification_manager.show_success(f"Opened project: {len(docs)} documents")

    def _init_mode_widgets(self) -> None:
        """Initialize mode-specific widgets with app context."""
        from lazylabeltext.ui.modes.chunk_mode import ChunkModeWidget
        from lazylabeltext.ui.modes.convert_mode import ConvertModeWidget
        from lazylabeltext.ui.modes.export_mode import ExportModeWidget
        from lazylabeltext.ui.modes.label_mode import LabelModeWidget
        from lazylabeltext.ui.modes.propagation_mode import PropagationModeWidget
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
            "propagation", PropagationModeWidget(self.app_context, self)
        )
        self.center_panel.set_mode_widget("export", ExportModeWidget(self.app_context))

    def _update_stats(self) -> None:
        """Update status bar statistics."""
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

    def _open_settings(self) -> None:
        dialog = ProviderSettingsDialog(self.settings, self)
        if dialog.exec():
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
        if self.settings.last_project_path:
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
