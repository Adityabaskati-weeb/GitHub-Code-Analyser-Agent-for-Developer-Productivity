"""
src/cache/disk_cache.py
=======================
Simple filesystem cache for GitHub API responses and raw file blobs.

Files are stored under CACHE_DIR (default: .cache/ at the project root).
Each entry is keyed by the SHA-256 of the original URL so filenames are
safe on all platforms.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from src.config.settings import CACHE_DIR

_cache_root = Path(CACHE_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cache_key(url: str) -> str:
    """Return a hex-digest that is safe to use as a filename component."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _path_for(url: str, suffix: str = ".txt") -> Path:
    return _cache_root / (cache_key(url) + suffix)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(url: str, suffix: str = ".txt") -> str | None:
    """
    Return the cached text for *url*, or ``None`` if not cached.
    """
    p = _path_for(url, suffix)
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return None
    return None


def load_json(url: str) -> dict | list | None:
    """Return cached JSON-decoded object, or ``None``."""
    raw = load(url, suffix=".json")
    if raw is not None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


def save(url: str, content: str, suffix: str = ".txt") -> None:
    """Persist *content* to disk under a key derived from *url*."""
    _cache_root.mkdir(parents=True, exist_ok=True)
    _path_for(url, suffix).write_text(content, encoding="utf-8")


def save_json(url: str, obj: dict | list) -> None:
    """Serialise *obj* as JSON and persist it."""
    save(url, json.dumps(obj, ensure_ascii=False), suffix=".json")


def invalidate(url: str, suffix: str = ".txt") -> None:
    """Delete a single cached entry (silently ignores missing files)."""
    p = _path_for(url, suffix)
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def clear() -> None:
    """Wipe the entire cache directory."""
    if _cache_root.exists():
        shutil.rmtree(_cache_root)
