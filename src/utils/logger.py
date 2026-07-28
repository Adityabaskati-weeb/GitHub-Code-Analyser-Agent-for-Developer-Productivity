"""
src/utils/logger.py
====================
Centralised logging configuration for the GitHub Code-Analyser Agent.

Usage
-----
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("fetching %s", url)

The root logger level is set once at import time from the ``LOG_LEVEL``
environment variable (default ``WARNING``).  The ``--verbose`` CLI flag
calls :func:`set_verbose` to lower it to ``DEBUG`` at runtime.

All agent output that should always be visible to the user (answers,
progress summaries) still goes through ``print()`` so it is never
suppressed by the logging level.
"""

from __future__ import annotations

import logging
import os
import sys

_ROOT = "code_analyser"

# Default level: WARNING (silent in normal use)
_DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()

logging.basicConfig(
    stream=sys.stderr,
    format="%(levelname)s [%(name)s] %(message)s",
    level=getattr(logging, _DEFAULT_LEVEL, logging.WARNING),
)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger scoped under the project root."""
    return logging.getLogger(f"{_ROOT}.{name}")


def set_verbose(verbose: bool = True) -> None:
    """Switch the root logger to DEBUG (verbose) or back to WARNING."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.getLogger(_ROOT).setLevel(level)
    # Also update the root handler so basicConfig level is respected
    logging.getLogger().setLevel(level)
