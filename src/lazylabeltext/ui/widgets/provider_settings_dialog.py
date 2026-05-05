"""Provider settings dialog for API key and model configuration."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class ProviderSettingsDialog(QDialog):
    """Dialog for configuring LLM and embedding providers."""

    def __init__(self, settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Provider Settings")
        self.setMinimumWidth(500)
        self._setup_ui()
        self._load_from_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # --- LLM Tab ---
        llm_tab = QWidget()
        llm_layout = QFormLayout(llm_tab)
        llm_layout.setSpacing(10)

        self.llm_provider_combo = QComboBox()
        self.llm_provider_combo.addItems(["anthropic", "openai", "google", "ollama"])
        self.llm_provider_combo.currentTextChanged.connect(
            self._on_llm_provider_changed
        )
        llm_layout.addRow("Provider:", self.llm_provider_combo)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText(
            "Enter API key or set ANTHROPIC_API_KEY env var"
        )

        key_row = QHBoxLayout()
        key_row.addWidget(self.api_key_edit)
        self.show_key_btn = QPushButton("Show")
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.toggled.connect(self._toggle_key_visibility)
        key_row.addWidget(self.show_key_btn)
        llm_layout.addRow("API Key:", key_row)

        self.llm_model_combo = QComboBox()
        self.llm_model_combo.setEditable(True)
        self.llm_model_combo.addItems(
            [
                "claude-sonnet-4-20250514",
                "claude-haiku-4-5-20251001",
            ]
        )
        llm_layout.addRow("Model:", self.llm_model_combo)

        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("http://localhost:11434 (for Ollama)")
        self.base_url_label = QLabel("Base URL:")
        llm_layout.addRow(self.base_url_label, self.base_url_edit)

        self.test_btn = QPushButton("Test Connection")
        self.test_btn.setObjectName("accentButton")
        self.test_btn.clicked.connect(self._test_llm_connection)
        self.test_result_label = QLabel()
        test_row = QHBoxLayout()
        test_row.addWidget(self.test_btn)
        test_row.addWidget(self.test_result_label, 1)
        llm_layout.addRow("", test_row)

        tabs.addTab(llm_tab, "LLM Provider")

        # --- Embedding Tab ---
        emb_tab = QWidget()
        emb_layout = QFormLayout(emb_tab)
        emb_layout.setSpacing(10)

        self.emb_provider_combo = QComboBox()
        self.emb_provider_combo.addItems(
            ["openai", "sentence-transformers", "none"]
        )
        self.emb_provider_combo.currentTextChanged.connect(
            self._on_emb_provider_changed
        )
        emb_layout.addRow("Provider:", self.emb_provider_combo)

        self.emb_model_combo = QComboBox()
        self.emb_model_combo.setEditable(True)
        emb_layout.addRow("Model:", self.emb_model_combo)

        self.emb_api_key_edit = QLineEdit()
        self.emb_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.emb_api_key_edit.setPlaceholderText(
            "Enter API key or set OPENAI_API_KEY env var"
        )
        emb_key_row = QHBoxLayout()
        emb_key_row.addWidget(self.emb_api_key_edit)
        self.emb_show_key_btn = QPushButton("Show")
        self.emb_show_key_btn.setCheckable(True)
        self.emb_show_key_btn.toggled.connect(self._toggle_emb_key_visibility)
        emb_key_row.addWidget(self.emb_show_key_btn)
        self.emb_api_key_label = QLabel("API Key:")
        emb_layout.addRow(self.emb_api_key_label, emb_key_row)

        emb_info = QLabel(
            "Embeddings power kNN agreement (a second confidence signal) and "
            "are persisted with each chunk for downstream RAG. "
            "Choose 'none' to disable both."
        )
        emb_info.setWordWrap(True)
        emb_info.setStyleSheet("color: #888; font-size: 11px;")
        emb_layout.addRow("", emb_info)

        tabs.addTab(emb_tab, "Embedding Provider")

        layout.addWidget(tabs)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        save_btn = QPushButton("Save")
        save_btn.setObjectName("accentButton")
        save_btn.clicked.connect(self._save_and_accept)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

        self._on_llm_provider_changed(self.llm_provider_combo.currentText())

    def _load_from_settings(self) -> None:
        self.llm_provider_combo.setCurrentText(self.settings.llm_provider)
        self.api_key_edit.setText(self.settings.llm_api_key)
        self.llm_model_combo.setCurrentText(self.settings.llm_model)
        self.base_url_edit.setText(self.settings.llm_base_url)
        self.emb_provider_combo.setCurrentText(self.settings.embedding_provider)
        self._on_emb_provider_changed(self.emb_provider_combo.currentText())
        self.emb_model_combo.setCurrentText(self.settings.embedding_model)
        self.emb_api_key_edit.setText(
            getattr(self.settings, "embedding_api_key", "")
        )

    def _on_llm_provider_changed(self, provider: str) -> None:
        is_ollama = provider == "ollama"
        self.base_url_edit.setVisible(is_ollama)
        self.base_url_label.setVisible(is_ollama)

        needs_key = provider in ("anthropic", "openai", "google")
        self.api_key_edit.setEnabled(needs_key)
        self.show_key_btn.setEnabled(needs_key)

        env_var_hint = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }.get(provider)
        if env_var_hint:
            self.api_key_edit.setPlaceholderText(
                f"Enter API key or set {env_var_hint} env var"
            )
        else:
            self.api_key_edit.setPlaceholderText(
                "(Ollama uses no API key — set Base URL below)"
            )

        # Update model suggestions
        self.llm_model_combo.clear()
        if provider == "anthropic":
            self.llm_model_combo.addItems(
                [
                    "claude-opus-4-7",
                    "claude-sonnet-4-6",
                    "claude-haiku-4-5-20251001",
                ]
            )
        elif provider == "openai":
            self.llm_model_combo.addItems(
                ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"]
            )
        elif provider == "google":
            self.llm_model_combo.addItems(
                ["gemini-2.5-flash", "gemini-2.5-pro"]
            )
        elif provider == "ollama":
            self.llm_model_combo.addItems(
                ["llama3", "mistral", "gemma2", "qwen2.5"]
            )

    def _on_emb_provider_changed(self, provider: str) -> None:
        needs_key = provider in ("openai", "openai-embeddings")
        self.emb_api_key_edit.setEnabled(needs_key)
        self.emb_show_key_btn.setEnabled(needs_key)
        self.emb_api_key_label.setEnabled(needs_key)

        is_none = provider in ("none", "disabled", "off", "")
        self.emb_model_combo.setEnabled(not is_none)

        self.emb_model_combo.clear()
        if provider in ("openai", "openai-embeddings"):
            self.emb_model_combo.addItems(
                [
                    "text-embedding-ada-002",
                    "text-embedding-3-small",
                    "text-embedding-3-large",
                ]
            )
        elif provider == "sentence-transformers":
            self.emb_model_combo.addItems(
                [
                    "all-MiniLM-L6-v2",
                    "all-mpnet-base-v2",
                    "BAAI/bge-small-en-v1.5",
                ]
            )
        else:
            self.emb_model_combo.addItem("(disabled)")

    def _toggle_emb_key_visibility(self, checked: bool) -> None:
        self.emb_api_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )
        self.emb_show_key_btn.setText("Hide" if checked else "Show")

    def _toggle_key_visibility(self, checked: bool) -> None:
        self.api_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )
        self.show_key_btn.setText("Hide" if checked else "Show")

    def _test_llm_connection(self) -> None:
        self.test_result_label.setText("Testing...")
        self.test_result_label.setStyleSheet("color: #888;")

        from lazylabeltext.core.providers import create_llm_provider

        provider_name = self.llm_provider_combo.currentText()
        kwargs: dict = {
            "model": self.llm_model_combo.currentText(),
        }
        if provider_name in ("anthropic", "openai", "google"):
            kwargs["api_key"] = self.api_key_edit.text()
        if provider_name == "ollama":
            base_url = self.base_url_edit.text().strip()
            if base_url:
                kwargs["base_url"] = base_url

        try:
            provider = create_llm_provider(provider_name, **kwargs)
            if provider is None:
                self.test_result_label.setText(
                    f"Could not initialize {provider_name} provider"
                )
                self.test_result_label.setStyleSheet("color: #ff6b6b;")
                return
            success, msg = provider.test_connection()
            color = "#51cf66" if success else "#ff6b6b"
            self.test_result_label.setText(msg)
            self.test_result_label.setStyleSheet(f"color: {color};")
        except Exception as e:
            self.test_result_label.setText(str(e))
            self.test_result_label.setStyleSheet("color: #ff6b6b;")

    def _save_and_accept(self) -> None:
        self.settings.llm_provider = self.llm_provider_combo.currentText()
        self.settings.llm_api_key = self.api_key_edit.text()
        self.settings.llm_model = self.llm_model_combo.currentText()
        self.settings.llm_base_url = self.base_url_edit.text()
        self.settings.embedding_provider = self.emb_provider_combo.currentText()
        self.settings.embedding_model = self.emb_model_combo.currentText()
        self.settings.embedding_api_key = self.emb_api_key_edit.text()
        self.accept()
