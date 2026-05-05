"""Tiny .env file loader.

Avoids the python-dotenv dependency — the .env format is simple enough that
50 lines of careful parsing covers what we need:

  KEY=value
  KEY="value with spaces"
  KEY='value with spaces'
  export KEY=value          # `export ` prefix is tolerated
  # comments are ignored
  blank lines are ignored

By default, an existing env var is NOT overwritten (so the user's shell
always wins). Pass `override=True` to flip that.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("lazylabeltext")


def load_dotenv_file(path: str | Path, override: bool = False) -> int:
    """Load KEY=value pairs from `path` into os.environ.

    Returns the number of variables actually set (i.e. not skipped because
    they were already in the env).
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return 0

    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Could not read .env file %s: %s", p, e)
        return 0

    set_count = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        # Strip matched surrounding quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        set_count += 1
    if set_count:
        logger.info(".env: loaded %d variable(s) from %s", set_count, p)
    return set_count
