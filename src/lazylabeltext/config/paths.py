"""Path management for LazyLabelText."""

from __future__ import annotations

import os
from pathlib import Path


class Paths:
    """Manages application directory paths."""

    def __init__(self) -> None:
        self.config_dir = Path(
            os.environ.get(
                "LLT_CONFIG_DIR",
                Path.home() / ".config" / "lazylabeltext",
            )
        )
        self.cache_dir = Path(
            os.environ.get(
                "LLT_CACHE_DIR",
                Path.home() / ".cache" / "lazylabeltext",
            )
        )
        self.log_dir = self.config_dir / "logs"

        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def settings_file(self) -> Path:
        return self.config_dir / "settings.json"

    @property
    def hotkeys_file(self) -> Path:
        return self.config_dir / "hotkeys.json"

    @property
    def log_file(self) -> Path:
        return self.log_dir / "lazylabeltext.log"

    @property
    def app_dir(self) -> Path:
        """Directory of the installed package (next to demo_pictures, etc.)."""
        return Path(__file__).resolve().parent.parent

    @property
    def demo_pictures_dir(self) -> Path:
        return self.app_dir / "demo_pictures"

    @property
    def logo_path(self) -> Path:
        return self.demo_pictures_dir / "logo2.png"

    @property
    def models_dir(self) -> Path:
        """Local model cache for offline-friendly embeddings.

        Mirrors LazyLabel's pattern: each model is downloaded once and saved
        here, then loaded from disk on subsequent runs. Letting users copy
        this folder onto an air-gapped machine is the manual-install path.
        """
        d = self.app_dir / "models"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Frozen / read-only install — caller will handle the missing dir.
            pass
        return d

    def model_path(self, model_name: str) -> Path:
        """Return the local path for a named embedding model."""
        # Slashes in HF model names ("BAAI/bge-small-en-v1.5") become subdirs.
        return self.models_dir / model_name.replace("/", "__")
