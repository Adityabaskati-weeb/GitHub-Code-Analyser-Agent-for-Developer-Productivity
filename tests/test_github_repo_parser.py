"""
tests/test_github_repo_parser.py
=================================
Unit tests for GitRepoParser URL parsing and default-branch logic.
All network calls are patched with unittest.mock.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from src.github_repo_parser import GitRepoParser


# ---------------------------------------------------------------------------
# _get_repo_name
# ---------------------------------------------------------------------------

class TestGetRepoName:

    def setup_method(self):
        # Instantiate without a token so no HTTP session header is set
        with patch("src.github_repo_parser.GITHUB_TOKEN", None):
            self.parser = GitRepoParser(github_token=False)

    def test_standard_url(self):
        assert self.parser._get_repo_name("https://github.com/owner/repo") == "owner/repo"

    def test_trailing_slash(self):
        assert self.parser._get_repo_name("https://github.com/owner/repo/") == "owner/repo"

    def test_git_suffix(self):
        assert self.parser._get_repo_name("https://github.com/owner/repo.git") == "owner/repo"

    def test_git_suffix_and_trailing_slash(self):
        assert self.parser._get_repo_name("https://github.com/owner/repo.git/") == "owner/repo"

    def test_leading_whitespace(self):
        assert self.parser._get_repo_name("  https://github.com/owner/repo  ") == "owner/repo"

    def test_missing_repo_part_raises_value_error(self):
        with pytest.raises(ValueError, match="must look like"):
            self.parser._get_repo_name("https://github.com/only-owner")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            self.parser._get_repo_name("https://github.com/")

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError, match="string"):
            self.parser._get_repo_name(12345)  # type: ignore[arg-type]

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            self.parser._get_repo_name(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _get_default_branch
# ---------------------------------------------------------------------------

class TestGetDefaultBranch:

    def setup_method(self):
        with patch("src.github_repo_parser.GITHUB_TOKEN", None):
            self.parser = GitRepoParser(github_token=False)

    def _make_response(self, default_branch: str, status_code: int = 200):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = {"default_branch": default_branch}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_returns_main(self):
        self.parser.s.get = MagicMock(return_value=self._make_response("main"))
        assert self.parser._get_default_branch("owner", "repo") == "main"

    def test_returns_master(self):
        self.parser.s.get = MagicMock(return_value=self._make_response("master"))
        assert self.parser._get_default_branch("owner", "repo") == "master"

    def test_returns_custom_branch(self):
        self.parser.s.get = MagicMock(return_value=self._make_response("develop"))
        assert self.parser._get_default_branch("owner", "repo") == "develop"

    def test_missing_key_falls_back_to_main(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}          # no "default_branch" key
        mock_resp.raise_for_status = MagicMock()
        self.parser.s.get = MagicMock(return_value=mock_resp)
        assert self.parser._get_default_branch("owner", "repo") == "main"

    def test_404_raises_value_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status.side_effect = Exception("404 Not Found")
        self.parser.s.get = MagicMock(return_value=mock_resp)
        with pytest.raises(Exception):
            self.parser._get_default_branch("owner", "nonexistent-repo")
