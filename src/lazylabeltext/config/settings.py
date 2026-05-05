"""Application settings with JSON persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Application settings persisted as JSON."""

    # Window
    window_width: int = 1600
    window_height: int = 900
    left_panel_width: int = 250
    right_panel_width: int = 350
    dark_mode: bool = True

    # Project
    last_project_path: str = ""

    # LLM Provider
    llm_provider: str = "anthropic"
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_base_url: str = ""

    # Embedding Provider — sentence-transformers (free, local) by default.
    # Switch to OpenAI ada-002 / text-embedding-3-* in Provider Settings if you want
    # a hosted model. "none" disables both kNN agreement and persisted embeddings.
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_api_key: str = ""

    # Labeling
    auto_accept_threshold: float = 0.0  # Disabled by default (conservative)
    multi_run_count: int = 1
    label_temperature: float = 0.3

    # Chunking defaults
    default_chunk_strategy: str = "structural"
    chunk_min_tokens: int = 50
    chunk_target_tokens: int = 300
    chunk_max_tokens: int = 800
    heading_split_levels: list[int] = field(default_factory=lambda: [1, 2, 3])

    # Export
    export_format: str = "json"

    def save_to_file(self, filepath: str | Path) -> None:
        """Save settings to JSON file."""
        data = asdict(self)
        # Never persist API keys to disk — they live in env vars or session state.
        data.pop("llm_api_key", None)
        data.pop("embedding_api_key", None)
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_from_file(cls, filepath: str | Path) -> Settings:
        """Load settings from JSON file, falling back to defaults."""
        filepath = Path(filepath)
        if not filepath.exists():
            return cls()
        try:
            with open(filepath) as f:
                data = json.load(f)
            # Only apply known fields
            known = {f.name for f in cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in data.items() if k in known}
            return cls(**filtered)
        except Exception:
            return cls()

    def update(self, **kwargs: object) -> None:
        """Update settings from keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


DEFAULT_SETTINGS = Settings()
