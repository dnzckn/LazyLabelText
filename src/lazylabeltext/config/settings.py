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
    # Doc map ingestion: alternate to opening a folder. When kind == "docmap"
    # the app loads source paths from a doc-map file and stores project.db
    # in last_project_output instead of inside the corpus folder.
    last_project_kind: str = "folder"  # "folder" | "docmap"
    last_docmap_path: str = ""
    last_project_output: str = ""

    # Conversion concurrency. 1 is safe for the high-fidelity (docling)
    # backend, which holds a singleton model. Lightweight backends
    # (PyMuPDF, python-docx, markdown) are happy at 4+.
    conversion_workers: int = 1

    # LLM Provider
    llm_provider: str = "anthropic"
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_base_url: str = ""
    # When True, the provider reads its API key (and Azure endpoint) from
    # environment variables — the in-app key field is hidden so secrets
    # never enter app config or settings.json.
    llm_use_env_credentials: bool = True

    # Optional .env file loader. Useful when shell-exported env vars don't
    # propagate to the launcher (Windows shortcuts, IDE terminals, etc.).
    # When enabled, KEY=value pairs in `dotenv_path` are loaded into
    # os.environ on startup *without* overwriting anything already set.
    dotenv_enabled: bool = False
    dotenv_path: str = ""  # blank → "<app_config_dir>/.env"

    # Azure OpenAI specifics — only consulted when llm_provider == "azure".
    # api_version + endpoint are typed in by the user, persisted between
    # sessions; auth still goes through env vars by default.
    llm_azure_api_version: str = "2024-08-01-preview"
    llm_azure_endpoint: str = ""
    llm_azure_verify_ssl: bool = True
    llm_azure_http2: bool = True

    # Embedding Provider — sentence-transformers (free, local) by default.
    # Switch to OpenAI ada-002 / text-embedding-3-* (or azure embeddings) in
    # Provider Settings. "none" disables kNN agreement + persisted embeddings.
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_api_key: str = ""

    # Labeling
    auto_accept_threshold: float = 0.0  # Disabled by default (conservative)
    multi_run_count: int = 1
    label_temperature: float = 0.3

    # High-fidelity conversion (docling). When enabled, .pdf and .docx are
    # routed through docling for proper table/figure structure. Slower and
    # downloads ~1–2 GB of models on first run. OCR is a separate switch
    # because it adds another model and is only needed for scanned PDFs.
    use_high_fidelity_conversion: bool = False
    high_fidelity_ocr: bool = False

    # Chunking defaults
    default_chunk_strategy: str = "structural"
    chunk_min_tokens: int = 50
    chunk_target_tokens: int = 300
    chunk_max_tokens: int = 800
    heading_split_levels: list[int] = field(default_factory=lambda: [1, 2, 3])

    # Parallel mode — default worker count for chunk/label stages.
    # Convert defaults to the same value but the user can dial it down via
    # the strip spinbox if running docling (which holds a heavy singleton
    # model and doesn't gain from many parallel converts).
    parallel_workers: int = 4

    # Max workers labeling chunks of the *same* doc concurrently. 1 keeps
    # the original "one worker per doc, sequential chunks within" behavior
    # so docs still finish in roughly first-finished-first order. Higher
    # lets a single doc finish faster (good when there's just one doc and
    # many idle workers); workers/max_collaborators is the upper bound on
    # how many distinct docs can be in flight at once.
    max_collaborators_per_doc: int = 1

    # Export
    export_format: str = "json"

    def save_to_file(self, filepath: str | Path) -> None:
        """Save settings to JSON file."""
        data = asdict(self)
        # Never persist API keys to disk — they live in env vars or session state.
        # Endpoint URLs are persisted (not secrets).
        data.pop("llm_api_key", None)
        data.pop("embedding_api_key", None)
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str | Path) -> Settings:
        """Load settings from JSON file, falling back to defaults."""
        filepath = Path(filepath)
        if not filepath.exists():
            return cls()
        try:
            with open(filepath, encoding="utf-8") as f:
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
