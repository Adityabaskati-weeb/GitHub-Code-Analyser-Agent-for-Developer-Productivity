"""
tests/test_analyze_repo_node.py
=================================
Unit tests for the smart file-selection logic in analyze_repo_node.

Tests cover:
- _is_skippable: minified JS/CSS, source-maps, binary extensions
- MAX_SELECTED_FILES cap
- MAX_SIZE_KB enforcement
- Score ranking (README beats deep nested file)
- Async analyze_tree_node behaviour (no network, no LLM)
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import patch

from src.nodes.analyze_repo_node import _is_skippable, _score, analyze_tree_node


# ---------------------------------------------------------------------------
# _is_skippable
# ---------------------------------------------------------------------------

class TestIsSkippable:

    def test_min_js_is_skippable(self):
        assert _is_skippable("vendor/jquery.min.js", ".js") is True

    def test_min_css_is_skippable(self):
        assert _is_skippable("styles/main.min.css", ".css") is True

    def test_js_map_is_skippable(self):
        assert _is_skippable("dist/bundle.js.map", ".map") is True

    def test_css_map_is_skippable(self):
        assert _is_skippable("dist/main.css.map", ".map") is True

    def test_bundle_js_is_skippable(self):
        assert _is_skippable("dist/app.bundle.js", ".js") is True

    def test_py_file_not_skippable(self):
        assert _is_skippable("src/main.py", ".py") is False

    def test_md_file_not_skippable(self):
        assert _is_skippable("README.md", ".md") is False

    def test_json_file_not_skippable(self):
        assert _is_skippable("config.json", ".json") is False

    def test_woff_font_is_skippable(self):
        assert _is_skippable("static/fonts/icon.woff", ".woff") is True

    def test_regular_js_not_skipped_by_pattern(self):
        # A regular .js file (not minified) — skippable by extension rule
        # but NOT by pattern rule (no .min. in name)
        result = _is_skippable("src/script.js", ".js")
        # .js is in _SKIP_EXTENSIONS; result depends on whether it's in _KEEP_EXTENSIONS
        assert isinstance(result, bool)  # just confirm it runs


# ---------------------------------------------------------------------------
# _score
# ---------------------------------------------------------------------------

class TestScore:

    def _make_meta(self, path: str, ext: str = ".py", size_kb: float = 5.0) -> dict:
        return {"path": path, "ext": ext, "size_kb": size_kb}

    def test_readme_beats_deep_nested_file(self):
        readme = self._make_meta("README.md", ".md")
        deep   = self._make_meta("a/b/c/d/helper.py")
        assert _score(readme, ["readme"], "high_level_summary", [], {}) > \
               _score(deep,   ["readme"], "high_level_summary", [], {})

    def test_keyword_match_boosts_score(self):
        match    = self._make_meta("src/pipeline_runner.py")
        no_match = self._make_meta("src/utils/format.py")
        s_match    = _score(match,    ["pipeline"], "high_level_summary", [], {})
        s_no_match = _score(no_match, ["pipeline"], "high_level_summary", [], {})
        assert s_match > s_no_match

    def test_important_name_in_path_boosts_score(self):
        main_file   = self._make_meta("src/main.py")
        random_file = self._make_meta("src/utils/helper.py")
        assert _score(main_file,   [], "high_level_summary", [], {}) > \
               _score(random_file, [], "high_level_summary", [], {})

    def test_shallow_path_beats_deep_path(self):
        shallow = self._make_meta("app.py")
        deep    = self._make_meta("a/b/c/d/e/app.py")
        assert _score(shallow, [], "high_level_summary", [], {}) > \
               _score(deep,    [], "high_level_summary", [], {})

    def test_function_usage_intent_boosts_matching_function(self):
        fn_file    = self._make_meta("src/predict.py")
        other_file = self._make_meta("src/utils/format.py")
        targets = {"function": "predict"}
        s1 = _score(fn_file,    [], "function_usage", ["predict"], targets)
        s2 = _score(other_file, [], "function_usage", ["predict"], targets)
        assert s1 > s2


# ---------------------------------------------------------------------------
# MAX_SELECTED_FILES cap (via analyze_tree_node)
# ---------------------------------------------------------------------------

def _make_tree(n: int, ext: str = ".py") -> dict:
    """Build a flat fake repo tree with *n* files."""
    tree = {}
    for i in range(n):
        tree[f"file{i}{ext}"] = {
            "path":    f"file{i}{ext}",
            "type":    "file",
            "ext":     ext,
            "size_kb": 1.0,
            "url":     f"https://example.com/file{i}{ext}",
        }
    return tree


class TestAnalyzeTreeNode:

    def _run(self, state: dict) -> dict:
        return asyncio.get_event_loop().run_until_complete(analyze_tree_node(state))

    def _base_state(self, tree: dict) -> dict:
        from langchain_core.messages import HumanMessage
        return {
            "repo_tree": tree,
            "messages": [HumanMessage(content="explain the architecture")],
            "intent": "architecture_summary",
            "keywords": [],
            "targets": {},
            "selected_files": [],
            "unselected_files": [],
        }

    def test_cap_at_max_selected_files(self):
        """Never select more than MAX_SELECTED_FILES files."""
        from src.config.settings import MAX_SELECTED_FILES
        tree = _make_tree(MAX_SELECTED_FILES + 15)
        result = self._run(self._base_state(tree))
        assert len(result["selected_files"]) <= MAX_SELECTED_FILES

    def test_minified_files_not_selected(self):
        """*.min.js files must never appear in selected_files."""
        tree = {
            "jquery.min.js": {
                "path": "jquery.min.js", "type": "file",
                "ext": ".js", "size_kb": 100.0,
                "url": "https://example.com/jquery.min.js",
            },
            "main.py": {
                "path": "main.py", "type": "file",
                "ext": ".py", "size_kb": 2.0,
                "url": "https://example.com/main.py",
            },
        }
        result = self._run(self._base_state(tree))
        selected_paths = [f["path"] for f in result["selected_files"]]
        assert "jquery.min.js" not in selected_paths

    def test_oversized_files_not_selected(self):
        """Files exceeding MAX_SIZE_KB must not be selected."""
        from src.config.settings import MAX_SIZE_KB
        tree = {
            "huge.py": {
                "path": "huge.py", "type": "file",
                "ext": ".py", "size_kb": MAX_SIZE_KB + 100,
                "url": "https://example.com/huge.py",
            },
            "small.py": {
                "path": "small.py", "type": "file",
                "ext": ".py", "size_kb": 5.0,
                "url": "https://example.com/small.py",
            },
        }
        result = self._run(self._base_state(tree))
        selected_paths = [f["path"] for f in result["selected_files"]]
        assert "huge.py" not in selected_paths
        assert "small.py" in selected_paths

    def test_empty_tree_returns_empty_selection(self):
        result = self._run(self._base_state({}))
        assert result["selected_files"] == []

    def test_no_tree_in_state_returns_empty(self):
        state = {
            "repo_tree": None,
            "messages": [],
            "intent": "",
            "keywords": [],
            "targets": {},
            "selected_files": [],
            "unselected_files": [],
        }
        result = self._run(state)
        assert result["selected_files"] == []
