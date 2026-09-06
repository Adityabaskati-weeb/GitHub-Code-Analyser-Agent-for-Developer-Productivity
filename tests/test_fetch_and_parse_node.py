"""
tests/test_fetch_and_parse_node.py
=====================================
Unit tests for concurrent fetch_and_parse_node.

All HTTP is mocked -- no real GitHub calls.
Tests cover:
- concurrent fetching with mocked fetch_blob_content
- deduplication (already-parsed files not re-fetched)
- empty selected_files short-circuit
- parse error handling
- TQDM_DISABLE respected
"""

from __future__ import annotations

import asyncio
import os
import pytest
from unittest.mock import patch, MagicMock

from src.nodes.fetch_and_parse_node import fetch_and_parse_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_file(path: str, ext: str = ".py", url: str = "https://x.com/f") -> dict:
    return {"path": path, "ext": ext, "url": url, "size_kb": 1.0}


def _run(coro):
    return asyncio.run(coro)


# Disable tqdm in all tests
os.environ["TQDM_DISABLE"] = "1"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFetchAndParseNode:

    def _base_state(self, selected=None, parsed=None):
        return {
            "selected_files": selected or [],
            "parsed_files":   parsed   or [],
            "refresh_cache": False,
            "messages": [],
        }

    def test_empty_selected_files_returns_empty(self):
        result = _run(fetch_and_parse_node(self._base_state()))
        assert result["parsed_files"] == []

    def test_fetches_and_parses_python_file(self):
        files = [_make_file("main.py", ".py")]
        code  = "import os\ndef hello(): pass\n"

        with patch("src.nodes.fetch_and_parse_node.fetch_blob_content", return_value=code):
            result = _run(fetch_and_parse_node(self._base_state(selected=files)))

        assert len(result["parsed_files"]) == 1
        pf = result["parsed_files"][0]
        assert pf["path"] == "main.py"
        assert "hello" in pf["parsed"]

    def test_already_parsed_file_is_not_refetched(self):
        already = [{"path": "main.py", "ext": ".py", "parsed": "existing content"}]
        files   = [_make_file("main.py", ".py")]

        fetch_mock = MagicMock(return_value="new content")
        with patch("src.nodes.fetch_and_parse_node.fetch_blob_content", fetch_mock):
            result = _run(fetch_and_parse_node(self._base_state(selected=files, parsed=already)))

        # fetch should NOT have been called
        fetch_mock.assert_not_called()
        # existing parsed file preserved
        assert result["parsed_files"][0]["parsed"] == "existing content"

    def test_multiple_files_fetched_concurrently(self):
        files = [_make_file(f"file{i}.py", ".py", f"https://x.com/f{i}") for i in range(5)]
        calls = []

        def fake_fetch(url, refresh=False):
            calls.append(url)
            return f"def fn_{url[-1]}(): pass\n"

        with patch("src.nodes.fetch_and_parse_node.fetch_blob_content", side_effect=fake_fetch):
            result = _run(fetch_and_parse_node(self._base_state(selected=files)))

        assert len(result["parsed_files"]) == 5
        assert len(calls) == 5

    def test_fetch_failure_recorded_as_error(self):
        files = [_make_file("broken.py")]

        with patch("src.nodes.fetch_and_parse_node.fetch_blob_content", return_value=""):
            result = _run(fetch_and_parse_node(self._base_state(selected=files)))

        assert len(result["parsed_files"]) == 1
        assert "Failed to fetch" in result["parsed_files"][0]["parsed"]

    def test_output_order_matches_input_order(self):
        """Results should be in the same order as selected_files."""
        files = [_make_file(f"f{i}.py", ".py", f"https://x.com/{i}") for i in range(6)]

        def fake_fetch(url, refresh=False):
            return f"# content for {url}\n"

        with patch("src.nodes.fetch_and_parse_node.fetch_blob_content", side_effect=fake_fetch):
            result = _run(fetch_and_parse_node(self._base_state(selected=files)))

        result_paths = [pf["path"] for pf in result["parsed_files"]]
        expected     = [f["path"] for f in files]
        assert result_paths == expected

    def test_unknown_extension_uses_raw_truncation(self):
        files = [_make_file("data.xyz", ".xyz")]

        with patch("src.nodes.fetch_and_parse_node.fetch_blob_content", return_value="raw content"):
            result = _run(fetch_and_parse_node(self._base_state(selected=files)))

        assert "raw content" in result["parsed_files"][0]["parsed"]
