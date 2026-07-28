"""
tests/test_flatten_tree.py
===========================
Unit tests for src.utils.flatten_tree.flatten_tree
"""

from __future__ import annotations

from src.utils.flatten_tree import flatten_tree


def _file(path: str, ext: str = ".py") -> dict:
    """Helper: build a minimal file metadata dict."""
    return {"path": path, "type": "file", "ext": ext, "size_kb": 1.0, "url": ""}


class TestFlattenTree:

    def test_empty_tree_returns_empty_list(self):
        assert flatten_tree({}) == []

    def test_single_top_level_file(self):
        tree = {"main.py": _file("main.py")}
        result = flatten_tree(tree)
        assert len(result) == 1
        assert result[0]["path"] == "main.py"

    def test_single_nested_file(self):
        tree = {
            "src/": {
                "app.py": _file("src/app.py"),
            }
        }
        result = flatten_tree(tree)
        assert len(result) == 1
        assert result[0]["path"] == "src/app.py"
        assert result[0]["folder"] == "src/"

    def test_multiple_files_in_same_folder(self):
        tree = {
            "src/": {
                "a.py": _file("src/a.py"),
                "b.py": _file("src/b.py"),
            }
        }
        paths = {f["path"] for f in flatten_tree(tree)}
        assert paths == {"src/a.py", "src/b.py"}

    def test_deeply_nested_structure(self):
        tree = {
            "src/": {
                "nodes/": {
                    "query_analyser_node.py": _file("src/nodes/query_analyser_node.py"),
                }
            }
        }
        result = flatten_tree(tree)
        assert len(result) == 1
        assert result[0]["path"] == "src/nodes/query_analyser_node.py"

    def test_mixed_depth(self):
        tree = {
            "README.md": _file("README.md", ".md"),
            "src/": {
                "app.py": _file("src/app.py"),
                "utils/": {
                    "helpers.py": _file("src/utils/helpers.py"),
                }
            }
        }
        paths = {f["path"] for f in flatten_tree(tree)}
        assert "README.md" in paths
        assert "src/app.py" in paths
        assert "src/utils/helpers.py" in paths
        assert len(paths) == 3

    def test_folder_without_files_returns_empty(self):
        tree = {"empty/": {}}
        assert flatten_tree(tree) == []

    def test_preserves_folder_field(self):
        tree = {
            "src/": {
                "app.py": _file("src/app.py"),
            }
        }
        result = flatten_tree(tree)
        assert result[0]["folder"] == "src/"

    def test_top_level_file_has_empty_folder(self):
        tree = {"main.py": _file("main.py")}
        result = flatten_tree(tree)
        assert result[0]["folder"] == ""

    def test_does_not_include_non_file_dicts(self):
        """Dicts without 'type':'file' should be treated as folders, not files."""
        tree = {
            "folder/": {
                # legitimate sub-dict (folder)
                "nested/": {},
                # file
                "real.py": _file("folder/real.py"),
            }
        }
        result = flatten_tree(tree)
        assert len(result) == 1
        assert result[0]["path"] == "folder/real.py"
