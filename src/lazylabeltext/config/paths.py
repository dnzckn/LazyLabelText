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
