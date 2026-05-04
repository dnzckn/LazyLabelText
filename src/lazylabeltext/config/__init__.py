"""Configuration management for LazyLabelText."""

from lazylabeltext.config.hotkeys import HotkeyAction, HotkeyManager
from lazylabeltext.config.paths import Paths
from lazylabeltext.config.settings import DEFAULT_SETTINGS, Settings

__all__ = [
    "HotkeyAction",
    "HotkeyManager",
    "Paths",
    "Settings",
    "DEFAULT_SETTINGS",
]
