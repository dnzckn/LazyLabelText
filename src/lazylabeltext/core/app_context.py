"""Dependency injection containers for LazyLabelText."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lazylabeltext.config.hotkeys import HotkeyManager
    from lazylabeltext.config.paths import Paths
    from lazylabeltext.config.settings import Settings
    from lazylabeltext.core.audit_manager import AuditManager
    from lazylabeltext.core.chunk_manager import ChunkManager
    from lazylabeltext.core.database import Database
    from lazylabeltext.core.document_manager import DocumentManager
    from lazylabeltext.core.label_manager import LabelManager
    from lazylabeltext.core.protocols import (
        EmbeddingProviderProtocol,
        LLMProviderProtocol,
    )
    from lazylabeltext.core.rubric_manager import RubricManager


@dataclass
class AppContext:
    """Holds references to all core managers and configuration."""

    paths: Paths | None = None
    settings: Settings | None = None
    hotkey_manager: HotkeyManager | None = None
    database: Database | None = None
    document_manager: DocumentManager | None = None
    rubric_manager: RubricManager | None = None
    chunk_manager: ChunkManager | None = None
    label_manager: LabelManager | None = None
    audit_manager: AuditManager | None = None
    llm_provider: LLMProviderProtocol | None = None
    embedding_provider: EmbeddingProviderProtocol | None = None

    _ui_state: dict[str, Any] = field(default_factory=dict)

    def set_ui_state(self, key: str, value: Any) -> None:
        self._ui_state[key] = value

    def get_ui_state(self, key: str, default: Any = None) -> Any:
        return self._ui_state.get(key, default)
