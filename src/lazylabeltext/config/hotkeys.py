"""Hotkey management with JSON persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HotkeyAction:
    """Represents a hotkey action with primary and optional secondary key."""

    name: str
    description: str
    primary_key: str
    secondary_key: str | None = None
    category: str = "General"


class HotkeyManager:
    """Manages application hotkeys with persistence."""

    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir)
        self.hotkeys_file = self.config_dir / "hotkeys.json"
        self.actions: dict[str, HotkeyAction] = {}
        self._initialize_defaults()
        self.load_hotkeys()

    def _initialize_defaults(self) -> None:
        """Set up default hotkey mappings."""
        defaults = [
            # Navigation
            HotkeyAction(
                "next_document", "Next Document", "Right", category="Navigation"
            ),
            HotkeyAction(
                "previous_document", "Previous Document", "Left", category="Navigation"
            ),
            # Modes
            HotkeyAction("convert_mode", "Convert Mode", "F1", category="Modes"),
            HotkeyAction("rubric_mode", "Rubric Mode", "F2", category="Modes"),
            HotkeyAction("chunk_mode", "Chunk Mode", "F3", category="Modes"),
            HotkeyAction("label_mode", "Label Mode", "F4", category="Modes"),
            HotkeyAction("results_mode", "Results Mode", "F5", category="Modes"),
            HotkeyAction(
                "parallel_mode", "Parallel Mode", "F6", category="Modes"
            ),
            HotkeyAction("export_mode", "Export Mode", "F7", category="Modes"),
            # Labeling
            HotkeyAction("accept_label", "Accept Label", "Space", category="Labeling"),
            HotkeyAction(
                "reject_label", "Reject / Correct Label", "C", category="Labeling"
            ),
            HotkeyAction("skip_chunk", "Skip Chunk", "S", category="Labeling"),
            HotkeyAction("flag_chunk", "Flag for Review", "F", category="Labeling"),
            HotkeyAction("add_note", "Add Note", "N", category="Labeling"),
            # Chunking
            HotkeyAction("merge_chunks", "Merge with Next", "M", category="Chunking"),
            HotkeyAction("split_chunk", "Split at Cursor", "T", category="Chunking"),
            # General
            HotkeyAction("save_project", "Save Project", "Ctrl+S", category="General"),
            HotkeyAction("open_folder", "Open Folder", "Ctrl+O", category="General"),
            HotkeyAction("settings", "Open Settings", "Ctrl+,", category="General"),
            HotkeyAction("toggle_theme", "Toggle Theme", "Ctrl+T", category="General"),
        ]
        for action in defaults:
            self.actions[action.name] = action

    def get_key_for_action(self, action_name: str) -> tuple[str | None, str | None]:
        """Get primary and secondary keys for an action."""
        action = self.actions.get(action_name)
        if action is None:
            return None, None
        return action.primary_key, action.secondary_key

    def get_actions_by_category(self) -> dict[str, list[HotkeyAction]]:
        """Get actions grouped by category."""
        categories: dict[str, list[HotkeyAction]] = {}
        for action in self.actions.values():
            categories.setdefault(action.category, []).append(action)
        return categories

    def save_hotkeys(self) -> None:
        """Save hotkeys to JSON file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        data = {}
        for name, action in self.actions.items():
            data[name] = {
                "primary_key": action.primary_key,
                "secondary_key": action.secondary_key,
            }
        with open(self.hotkeys_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load_hotkeys(self) -> None:
        """Load hotkeys from JSON file, keeping defaults for missing actions."""
        if not self.hotkeys_file.exists():
            return
        try:
            with open(self.hotkeys_file, encoding="utf-8") as f:
                data = json.load(f)
            for name, keys in data.items():
                if name in self.actions:
                    self.actions[name].primary_key = keys.get(
                        "primary_key", self.actions[name].primary_key
                    )
                    self.actions[name].secondary_key = keys.get("secondary_key")
        except Exception:
            pass

    def reset_to_defaults(self) -> None:
        """Reset all hotkeys to defaults."""
        self.actions.clear()
        self._initialize_defaults()
