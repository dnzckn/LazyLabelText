"""Styled logging setup for LazyLabelText."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


class _StyledFormatter(logging.Formatter):
    """Console formatter with ANSI colors."""

    COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self.use_color:
            color = self.COLORS.get(record.levelno, "")
            return f"{color}{msg}{self.RESET}"
        return msg


def setup_logging(log_file: str | Path | None = None) -> logging.Logger:
    """Configure logging for the application."""
    logger = logging.getLogger("lazylabeltext")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Console handler with color
    use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_StyledFormatter(use_color=use_color))
    logger.addHandler(console)

    # File handler (plain text)
    if log_file is None:
        from lazylabeltext.config.paths import Paths

        log_file = Paths().log_file

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    return logger
