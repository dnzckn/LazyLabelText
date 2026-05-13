"""Provider settings dialog for API key and model configuration."""

from __future__ import annotations

from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
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

    def showEvent(self, event: QShowEvent) -> None:
        """Center the dialog on its parent window when first shown."""
        super().showEvent(event)
        parent = self.parent()
        if parent is not None and hasattr(parent, "geometry"):
            try:
                pg = parent.geometry()
                self.move(
                    pg.x() + (pg.width() - self.width()) // 2,
                    pg.y() + (pg.height() - self.height()) // 2,
                )
            except Exception:
                pass

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # --- LLM Tab ---
        llm_tab = QWidget()
        llm_layout = QFormLayout(llm_tab)
        llm_layout.setSpacing(10)

        self.llm_provider_combo = QComboBox()
        self.llm_provider_combo.addItems(
            ["anthropic", "openai", "google", "azure", "ollama"]
        )
        self.llm_provider_combo.currentTextChanged.connect(
            self._on_llm_provider_changed
        )
        llm_layout.addRow("Provider:", self.llm_provider_combo)

        self.use_env_check = QCheckBox(
            "Use environment variables for credentials"
        )
        self.use_env_check.setToolTip(
            "When checked, the provider reads its API key (and Azure endpoint) "
            "from environment variables — secrets never touch the in-app field "
            "or settings.json."
        )
        self.use_env_check.toggled.connect(self._on_use_env_toggled)
        llm_layout.addRow("", self.use_env_check)

        # .env file loader — fallback when shell env doesn't propagate to the
        # launcher (Windows shortcuts, IDE terminals, frozen builds, etc.).
        self.dotenv_check = QCheckBox("Load .env file at startup")
        self.dotenv_check.setToolTip(
            "Read KEY=value pairs from a .env file and inject them into the "
            "process environment on startup. Existing env vars are NEVER "
            "overwritten — your shell still wins. The file is read once at "
            "launch; restart the app after editing it."
        )
        self.dotenv_check.toggled.connect(self._on_dotenv_toggled)
        llm_layout.addRow("", self.dotenv_check)

        self.dotenv_path_edit = QLineEdit()
        self.dotenv_path_edit.setPlaceholderText(
            "(defaults to <config-dir>/.env)"
        )
        self.dotenv_browse_btn = QPushButton("Browse…")
        self.dotenv_browse_btn.clicked.connect(self._browse_dotenv_path)
        dotenv_row = QHBoxLayout()
        dotenv_row.addWidget(self.dotenv_path_edit)
        dotenv_row.addWidget(self.dotenv_browse_btn)
        self.dotenv_path_label = QLabel(".env path:")
        llm_layout.addRow(self.dotenv_path_label, dotenv_row)

        dotenv_help = QLabel(
            "See .env.example at the repo root for the format and the full "
            "list of supported keys. Existing shell env vars are never "
            "overwritten."
        )
        dotenv_help.setStyleSheet("color: #888; font-size: 11px;")
        dotenv_help.setWordWrap(True)
        llm_layout.addRow("", dotenv_help)

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
        self.api_key_label = QLabel("API Key:")
        llm_layout.addRow(self.api_key_label, key_row)

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

        # --- Azure-specific fields (visible only when provider == 'azure') ---
        self.azure_endpoint_edit = QLineEdit()
        self.azure_endpoint_edit.setPlaceholderText(
            "https://<resource>.openai.azure.com (or set AZURE_OPENAI_ENDPOINT)"
        )
        self.azure_endpoint_label = QLabel("Azure endpoint:")
        llm_layout.addRow(self.azure_endpoint_label, self.azure_endpoint_edit)

        self.azure_api_version_edit = QLineEdit()
        self.azure_api_version_edit.setPlaceholderText("e.g. 2024-08-01-preview")
        self.azure_api_version_label = QLabel("API version:")
        llm_layout.addRow(self.azure_api_version_label, self.azure_api_version_edit)

        self.azure_verify_check = QCheckBox("Verify SSL")
        self.azure_verify_check.setToolTip(
            "Uncheck to skip server-certificate validation on outbound HTTPS."
        )
        self.azure_http2_check = QCheckBox("HTTP/2")
        azure_http_row = QHBoxLayout()
        azure_http_row.addWidget(self.azure_verify_check)
        azure_http_row.addWidget(self.azure_http2_check)
        azure_http_row.addStretch()
        self.azure_http_label = QLabel("HTTP options:")
        llm_layout.addRow(self.azure_http_label, azure_http_row)

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
            ["openai", "azure", "sentence-transformers", "none"]
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
        self.use_env_check.setChecked(
            getattr(self.settings, "llm_use_env_credentials", True)
        )
        self.dotenv_check.setChecked(
            getattr(self.settings, "dotenv_enabled", False)
        )
        self.dotenv_path_edit.setText(
            getattr(self.settings, "dotenv_path", "")
        )
        self._on_dotenv_toggled(self.dotenv_check.isChecked())
        self.api_key_edit.setText(self.settings.llm_api_key)
        self.llm_model_combo.setCurrentText(self.settings.llm_model)
        self.base_url_edit.setText(self.settings.llm_base_url)

        # Azure-specific
        self.azure_endpoint_edit.setText(
            getattr(self.settings, "llm_azure_endpoint", "")
        )
        self.azure_api_version_edit.setText(
            getattr(self.settings, "llm_azure_api_version", "2024-08-01-preview")
        )
        self.azure_verify_check.setChecked(
            getattr(self.settings, "llm_azure_verify_ssl", True)
        )
        self.azure_http2_check.setChecked(
            getattr(self.settings, "llm_azure_http2", True)
        )

        self.emb_provider_combo.setCurrentText(self.settings.embedding_provider)
        self._on_emb_provider_changed(self.emb_provider_combo.currentText())
        self.emb_model_combo.setCurrentText(self.settings.embedding_model)
        self.emb_api_key_edit.setText(
            getattr(self.settings, "embedding_api_key", "")
        )

        # Apply visibility/state once everything is populated.
        self._on_llm_provider_changed(self.llm_provider_combo.currentText())
        self._on_use_env_toggled(self.use_env_check.isChecked())

    def _on_llm_provider_changed(self, provider: str) -> None:
        is_ollama = provider == "ollama"
        is_azure = provider == "azure"
        self.base_url_edit.setVisible(is_ollama)
        self.base_url_label.setVisible(is_ollama)

        # Auto-pair embedding with Azure when the LLM is Azure: same endpoint,
        # same auth, same SSL/HTTP config. ada-002 is the canonical default
        # deployment name; user can override after.
        if is_azure and self.emb_provider_combo.currentText() != "azure":
            self.emb_provider_combo.setCurrentText("azure")
            self.emb_model_combo.setCurrentText("text-embedding-ada-002")

        # Azure fields are only relevant for the azure provider.
        for w in (
            self.azure_endpoint_edit,
            self.azure_endpoint_label,
            self.azure_api_version_edit,
            self.azure_api_version_label,
            self.azure_verify_check,
            self.azure_http2_check,
            self.azure_http_label,
        ):
            w.setVisible(is_azure)

        needs_key = provider in ("anthropic", "openai", "google", "azure")
        self.api_key_label.setVisible(needs_key)
        for w in (self.api_key_edit, self.show_key_btn):
            w.setEnabled(needs_key)

        env_var_hint = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
        }.get(provider)
        if env_var_hint:
            self.api_key_edit.setPlaceholderText(
                f"Enter API key or set {env_var_hint} env var"
            )
        else:
            self.api_key_edit.setPlaceholderText(
                "(Ollama uses no API key — set Base URL below)"
            )

        # Re-apply env-var visibility now that the provider has changed.
        self._on_use_env_toggled(self.use_env_check.isChecked())

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
        elif provider == "azure":
            # In Azure, "model" is the deployment name. Suggest common names
            # but make the combo editable since users name deployments freely.
            self.llm_model_combo.addItems(
                ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
            )
        elif provider == "ollama":
            self.llm_model_combo.addItems(
                ["llama3", "mistral", "gemma2", "qwen2.5"]
            )

    def _on_dotenv_toggled(self, checked: bool) -> None:
        self.dotenv_path_edit.setEnabled(checked)
        self.dotenv_browse_btn.setEnabled(checked)
        self.dotenv_path_label.setEnabled(checked)
        # Enabling .env loader implies env-mode for credentials (the user is
        # explicitly delivering keys via env). Force the env-mode toggle on
        # so the credential fields hide / gray accordingly.
        if checked and not self.use_env_check.isChecked():
            self.use_env_check.setChecked(True)
        # Re-evaluate field visibility (covers both toggles changing).
        self._on_use_env_toggled(self.use_env_check.isChecked())

    def _browse_dotenv_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose .env file",
            self.dotenv_path_edit.text() or "",
            "Env files (*.env *);;All files (*)",
        )
        if path:
            self.dotenv_path_edit.setText(path)

    def _external_credentials_active(self) -> bool:
        # Either the env-vars toggle or the .env loader counts as
        # "credentials are delivered via os.environ" — both end up putting
        # keys into env vars that the provider reads.
        return (
            self.use_env_check.isChecked() or self.dotenv_check.isChecked()
        )

    def _on_use_env_toggled(self, checked: bool) -> None:
        provider = self.llm_provider_combo.currentText()
        # When credentials come from env vars (either OS-level env or a loaded
        # .env), hide the key input so the user can't accidentally save a secret.
        provider_uses_key = provider in ("anthropic", "openai", "google", "azure")
        env_mode = self._external_credentials_active()
        show_key = provider_uses_key and not env_mode
        self.api_key_edit.setVisible(show_key)
        self.show_key_btn.setVisible(show_key)
        self.api_key_label.setVisible(provider_uses_key)
        if provider_uses_key:
            if env_mode:
                env_var_hint = {
                    "anthropic": "ANTHROPIC_API_KEY",
                    "openai": "OPENAI_API_KEY",
                    "google": "GOOGLE_API_KEY",
                    "azure": "AZURE_OPENAI_API_KEY (+ AZURE_OPENAI_ENDPOINT)",
                }.get(provider, "")
                self.api_key_label.setText(f"API Key: (from {env_var_hint})")
            else:
                self.api_key_label.setText("API Key:")

        # Azure endpoint is part of the credential bundle — it's read from
        # AZURE_OPENAI_ENDPOINT in env-mode, so disable the input to match.
        # API version / SSL / HTTP2 are connection config, not credentials,
        # so they stay editable regardless.
        if provider == "azure":
            self.azure_endpoint_edit.setEnabled(not env_mode)
            self.azure_endpoint_label.setEnabled(not env_mode)
            if env_mode:
                self.azure_endpoint_label.setText(
                    "Azure endpoint: (from AZURE_OPENAI_ENDPOINT)"
                )
                self.azure_endpoint_edit.setPlaceholderText(
                    "(read from AZURE_OPENAI_ENDPOINT env var)"
                )
            else:
                self.azure_endpoint_label.setText("Azure endpoint:")
                self.azure_endpoint_edit.setPlaceholderText(
                    "https://<resource>.openai.azure.com (or set AZURE_OPENAI_ENDPOINT)"
                )

        # The embedding tab inherits the same env-mode toggle: env-mode on →
        # hide the embedding key field; the provider reads OPENAI_API_KEY (or
        # the shared Azure auth) from env vars same as the LLM does.
        self._sync_embedding_key_visibility()

    def _sync_embedding_key_visibility(self) -> None:
        emb_provider = self.emb_provider_combo.currentText()
        emb_uses_key = emb_provider in (
            "openai",
            "openai-embeddings",
            "azure",
            "azure-embeddings",
        )
        env_mode = self._external_credentials_active()
        show_emb_key = emb_uses_key and not env_mode
        self.emb_api_key_edit.setVisible(show_emb_key)
        self.emb_show_key_btn.setVisible(show_emb_key)
        # Keep the label visible to explain what's going on, but reword it.
        self.emb_api_key_label.setVisible(emb_uses_key)
        if emb_uses_key:
            if env_mode:
                env_hint = (
                    "AZURE_OPENAI_API_KEY"
                    if emb_provider in ("azure", "azure-embeddings")
                    else "OPENAI_API_KEY"
                )
                self.emb_api_key_label.setText(f"API Key: (from {env_hint})")
            else:
                self.emb_api_key_label.setText("API Key:")

    def _on_emb_provider_changed(self, provider: str) -> None:
        is_none = provider in ("none", "disabled", "off", "")
        self.emb_model_combo.setEnabled(not is_none)
        # Visibility of the key field is owned by _sync_embedding_key_visibility,
        # so it reflects both the provider AND the env-mode toggle on the
        # LLM tab.
        self._sync_embedding_key_visibility()

        self.emb_model_combo.clear()
        if provider in ("openai", "openai-embeddings") or provider in ("azure", "azure-embeddings"):
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
        use_env = self._external_credentials_active()
        kwargs: dict = {
            "model": self.llm_model_combo.currentText(),
        }
        if provider_name in ("anthropic", "openai", "google", "azure"):
            kwargs["api_key"] = (
                "" if use_env else self.api_key_edit.text()
            )
        if provider_name == "ollama":
            base_url = self.base_url_edit.text().strip()
            if base_url:
                kwargs["base_url"] = base_url
        if provider_name == "azure":
            kwargs["api_version"] = self.azure_api_version_edit.text().strip()
            endpoint = self.azure_endpoint_edit.text().strip()
            if endpoint:
                kwargs["azure_endpoint"] = endpoint
            kwargs["verify_ssl"] = self.azure_verify_check.isChecked()
            kwargs["http2"] = self.azure_http2_check.isChecked()
            kwargs["use_env_credentials"] = use_env

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
        self.settings.dotenv_enabled = self.dotenv_check.isChecked()
        self.settings.dotenv_path = self.dotenv_path_edit.text().strip()
        # "External credentials" = either the env-vars toggle or the .env loader
        # is on. The runtime providers gate "read from os.environ" on
        # llm_use_env_credentials, and a .env file populates os.environ at
        # startup — so for the providers' purposes the two toggles are the same
        # signal. Persist the OR'd value so dotenv-only configs work.
        env_mode = self._external_credentials_active()
        self.settings.llm_use_env_credentials = env_mode
        # Don't keep the field's text in memory if creds come from env vars or
        # a .env file — that way toggling env-mode off in the same session
        # doesn't accidentally reuse a stale paste.
        self.settings.llm_api_key = (
            "" if env_mode else self.api_key_edit.text()
        )
        self.settings.llm_model = self.llm_model_combo.currentText()
        self.settings.llm_base_url = self.base_url_edit.text()
        # Azure-specific (always saved; harmless when the provider isn't azure)
        self.settings.llm_azure_endpoint = self.azure_endpoint_edit.text().strip()
        self.settings.llm_azure_api_version = (
            self.azure_api_version_edit.text().strip() or "2024-08-01-preview"
        )
        self.settings.llm_azure_verify_ssl = self.azure_verify_check.isChecked()
        self.settings.llm_azure_http2 = self.azure_http2_check.isChecked()
        self.settings.embedding_provider = self.emb_provider_combo.currentText()
        self.settings.embedding_model = self.emb_model_combo.currentText()
        # Drop the embedding key when env-mode is on, same as the LLM key.
        self.settings.embedding_api_key = (
            "" if env_mode else self.emb_api_key_edit.text()
        )
        self.accept()
