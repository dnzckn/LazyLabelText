"""Animated ASCII art startup display for the console.

Modelled on LazyLabel's startup display — same animation/progress/tip flow,
adapted for LazyLabelText.
"""

from __future__ import annotations

import os
import random
import re
import shutil
import sys
import time

from lazylabeltext import __version__

# ANSI escape codes
_CLEAR = "\033[2J\033[3J\033[H"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_CYAN = "\033[36m"
_BRIGHT_CYAN = "\033[96m"
_GREEN = "\033[32m"
_BRIGHT_GREEN = "\033[92m"
_YELLOW = "\033[33m"
_WHITE = "\033[97m"
_GRAY = "\033[90m"

_LOGO = (
    "██╗      █████╗ ███████╗██╗   ██╗██╗      █████╗ ██████╗ ███████╗██╗  ████████╗███████╗██╗  ██╗████████╗\n"
    "██║     ██╔══██╗╚══███╔╝╚██╗ ██╔╝██║     ██╔══██╗██╔══██╗██╔════╝██║  ╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝\n"
    "██║     ███████║  ███╔╝  ╚████╔╝ ██║     ███████║██████╔╝█████╗  ██║     ██║   █████╗   ╚███╔╝    ██║   \n"
    "██║     ██╔══██║ ███╔╝    ╚██╔╝  ██║     ██╔══██║██╔══██╗██╔══╝  ██║     ██║   ██╔══╝   ██╔██╗    ██║   \n"
    "███████╗██║  ██║███████╗   ██║   ███████╗██║  ██║██████╔╝███████╗███████╗██║   ███████╗██╔╝ ██╗   ██║   \n"
    "╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚══════╝╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   "
)

_AUTHOR = "Deniz N. Cakan"
_TAGLINE = "AI-Assisted Text Labeling"
_TOTAL_STEPS = 5

_TIPS = [
    "Tip: Press F1-F7 to switch modes: Convert, Rubric, Chunk, Label, Results, Propagate, Export.",
    "Tip: Space accepts the current label, C corrects, S skips, F flags, D discards.",
    "Tip: Left and Right arrows navigate the chunk timeline in the Label tab.",
    "Tip: Click any cell in the timeline to jump to that chunk — accepted ones stay visible.",
    "Tip: Use M to merge adjacent chunks, T to split at cursor.",
    "Tip: The rubric is versioned — every save creates a new version, old labels stay tagged.",
    "Tip: Set ANTHROPIC_API_KEY to enable LLM labeling, or configure it in Provider Settings.",
    "Tip: Export includes a manifest with corpus statistics for reproducibility.",
    "Tip: All state changes are logged to the audit trail automatically.",
    "Tip: The Coverage map flags dead, low-confidence, and high-disagreement categories.",
    "Tip: Re-running chunking on a doc replaces all prior chunks and labels for that doc.",
    "Tip: Reset Project (left panel) wipes project.db; source files on disk are kept.",
    "Tip: Hybrid chunking does structural splits first, then semantic on oversized chunks.",
    "Tip: LLM chunking sends each window to Claude for atomic-clause splits — slower but most accurate.",
    "Tip: Manual corrections show as 100% confidence in the Results table.",
    "Tip: Propagation (F6) batch-applies chunking + labeling to every document.",
]


def _is_tty() -> bool:
    """Check if stdout is a real terminal (not devnull / piped / PyInstaller hidden)."""
    try:
        return (
            sys.stdout is not None
            and hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
        )
    except Exception:
        return False


class StartupDisplay:
    """Animated console startup display with ASCII art and progress bar."""

    def __init__(self) -> None:
        self._enabled = _is_tty()
        self._width = shutil.get_terminal_size((80, 24)).columns
        self._logo_lines = _LOGO.split("\n")
        self._logo_width = max(len(line) for line in self._logo_lines)
        self._real_stdout: object | None = None
        self._real_stderr: object | None = None

    def _center(self, text: str) -> str:
        """Center a line of text based on its visible (non-ANSI) length."""
        visible = re.sub(r"\033\[[0-9;]*m", "", text)
        padding = max(0, (self._width - len(visible)) // 2)
        return " " * padding + text

    def _write(self, *parts: str) -> None:
        target = self._real_stdout or sys.stdout
        target.write("".join(parts))
        target.flush()

    def _draw_frame(self, step: int, message: str, *, final: bool = False) -> None:
        """Clear screen and redraw the full display for one frame."""
        lines: list[str] = [_CLEAR, ""]

        logo_color = _BRIGHT_GREEN if final else _BRIGHT_CYAN
        logo_pad = max(0, (self._width - self._logo_width) // 2)
        for logo_line in self._logo_lines:
            padded = logo_line.ljust(self._logo_width)
            lines.append(f"{' ' * logo_pad}{logo_color}{_BOLD}{padded}{_RESET}")

        lines.append("")
        lines.append(self._center(f"{_WHITE}{_BOLD}{_AUTHOR}{_RESET}"))
        lines.append(self._center(f"{_CYAN}{_TAGLINE}  {_WHITE}v{__version__}{_RESET}"))
        lines.append("")
        lines.append("")

        bar_width = min(40, self._width - 20)
        filled = int(bar_width * step / _TOTAL_STEPS)
        empty = bar_width - filled

        if final:
            bar = f"{_BRIGHT_GREEN}{'━' * bar_width}{_RESET}"
        else:
            bar = f"{_GREEN}{'━' * filled}{_GRAY}{'─' * empty}{_RESET}"

        pct = int(100 * step / _TOTAL_STEPS)
        lines.append(self._center(f"  {bar}  {_WHITE}{pct:>3}%{_RESET}"))
        lines.append("")

        if final:
            status = f"{_BRIGHT_GREEN}{_BOLD}>>> {message}{_RESET}"
        else:
            dots = "." * ((step % 3) + 1)
            status = f"{_YELLOW}>>> {message}{dots}{_RESET}"
        lines.append(self._center(status))
        lines.append("")

        self._write("\n".join(lines))

    def _capture_output(self) -> None:
        """Redirect stdout, stderr, and all logging StreamHandlers to devnull."""
        import logging

        self._real_stdout = sys.stdout
        self._real_stderr = sys.stderr
        self._saved_streams: list[tuple[logging.StreamHandler, object]] = []
        devnull = open(os.devnull, "w")  # noqa: SIM115
        sys.stdout = devnull
        sys.stderr = devnull

        for lg_name in [None, "lazylabeltext"]:
            for handler in logging.getLogger(lg_name).handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(
                    handler, logging.FileHandler
                ):
                    self._saved_streams.append((handler, handler.stream))
                    handler.stream = devnull

    def _release_output(self) -> None:
        """Restore stdout, stderr, and logging StreamHandler streams."""
        if self._real_stdout is not None:
            devnull = sys.stdout
            sys.stdout = self._real_stdout
            sys.stderr = self._real_stderr
            self._real_stdout = None
            self._real_stderr = None

            for handler, original_stream in self._saved_streams:
                handler.stream = original_stream
            self._saved_streams.clear()

            devnull.close()

    def show_banner(self) -> None:
        """Reveal the ASCII art logo line by line with animation."""
        if not self._enabled:
            return

        self._capture_output()
        self._write(_CLEAR)

        logo_pad = max(0, (self._width - self._logo_width) // 2)
        for i in range(len(self._logo_lines)):
            frame: list[str] = [_CLEAR, ""]
            for j in range(i + 1):
                padded = self._logo_lines[j].ljust(self._logo_width)
                frame.append(f"{' ' * logo_pad}{_BRIGHT_CYAN}{_BOLD}{padded}{_RESET}")
            self._write("\n".join(frame))
            time.sleep(0.05)

        time.sleep(0.12)
        self._draw_frame(0, "Starting up")

    def update_step(self, step: int, message: str) -> None:
        """Redraw the display with an updated progress bar and status."""
        if not self._enabled:
            return
        self._draw_frame(step, message)

    def finish(self) -> None:
        """Show the final 'ready' state and hand control back to normal output."""
        if not self._enabled:
            return
        self._draw_frame(
            _TOTAL_STEPS,
            random.choice(_TIPS),
            final=True,  # noqa: S311
        )
        time.sleep(0.25)
        self._write("\n\n")
        self._release_output()


# Module-level singleton
startup_display = StartupDisplay()


# Backwards-compat shim for older callers
def show_banner() -> None:
    startup_display.show_banner()
    startup_display.finish()
