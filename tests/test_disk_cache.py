"""
tests/test_disk_cache.py
==========================
Unit tests for src.cache.disk_cache.

Tests cover:
- cache miss (load returns None for unknown URL)
- cache hit (load returns content after save)
- save/load round-trip for text and JSON
- invalidate removes a single entry
- clear wipes the entire cache directory
- cache_key is deterministic and filesystem-safe
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers – patch CACHE_DIR so tests never touch the real .cache/ folder
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_cache(tmp_path):
    """Redirect all cache operations to a throwaway temp directory."""
    with patch("src.cache.disk_cache._cache_root", tmp_path):
        yield tmp_path


# Lazy import AFTER patching so the module picks up the temp root
from src.cache.disk_cache import (
    cache_key, load, save, load_json, save_json, invalidate, clear,
)


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:

    def test_same_url_returns_same_key(self):
        url = "https://raw.githubusercontent.com/owner/repo/main/README.md"
        assert cache_key(url) == cache_key(url)

    def test_different_urls_return_different_keys(self):
        assert cache_key("https://a.com/x") != cache_key("https://b.com/y")

    def test_key_is_hex_string(self):
        key = cache_key("https://example.com/file.py")
        assert re.fullmatch(r"[0-9a-f]{64}", key), f"Not a hex string: {key}"

    def test_key_contains_no_path_separators(self):
        key = cache_key("https://github.com/owner/repo/blob/main/deep/path.py")
        assert "/" not in key and "\\" not in key


# ---------------------------------------------------------------------------
# load / save (text)
# ---------------------------------------------------------------------------

class TestLoadSave:

    def test_load_unknown_url_returns_none(self):
        assert load("https://not-cached.com/file.py") is None

    def test_save_and_load_round_trip(self):
        url     = "https://example.com/main.py"
        content = "def hello(): return 'world'\n"
        save(url, content)
        assert load(url) == content

    def test_load_after_invalidate_returns_none(self):
        url = "https://example.com/file.py"
        save(url, "content")
        invalidate(url)
        assert load(url) is None

    def test_invalidate_missing_file_does_not_raise(self):
        invalidate("https://not-cached.com/missing.py")  # should be silent


# ---------------------------------------------------------------------------
# load_json / save_json
# ---------------------------------------------------------------------------

class TestJsonCache:

    def test_save_json_and_load_json_round_trip(self):
        url  = "https://api.github.com/repos/owner/repo"
        data = {"default_branch": "main", "files": [1, 2, 3]}
        save_json(url, data)
        loaded = load_json(url)
        assert loaded == data

    def test_load_json_unknown_url_returns_none(self):
        assert load_json("https://not-cached.com/tree.json") is None

    def test_load_json_corrupt_data_returns_none(self):
        url = "https://example.com/corrupt"
        # Save invalid JSON bytes directly
        from src.cache.disk_cache import _path_for
        p = _path_for(url, ".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{ not valid json }", encoding="utf-8")
        assert load_json(url) is None


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestClear:

    def test_clear_removes_all_cached_files(self, tmp_path):
        save("https://a.com/1", "one")
        save("https://b.com/2", "two")
        clear()
        assert not tmp_path.exists() or not list(tmp_path.iterdir())

    def test_clear_on_empty_cache_does_not_raise(self):
        clear()   # Should not raise even if already empty
        clear()   # Idempotent
