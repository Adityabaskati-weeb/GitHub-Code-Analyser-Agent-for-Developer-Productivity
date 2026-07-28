"""
Script to parse Repository and store contents
in textual form.
"""

from __future__ import annotations

import json
import os
import requests

from src.cache.disk_cache import load_json as cache_load_json, save_json as cache_save_json
from src.config.settings import GITHUB_TOKEN, EXCLUDE_EXT


class GitRepoParser:
    def __init__(self, github_token: bool = True):
        self.s = requests.Session()
        if github_token and GITHUB_TOKEN:
            self.s.headers.update({"Authorization": f"Token {GITHUB_TOKEN}"})

        self.exclude_ext = EXCLUDE_EXT

        # Raw API (for fetching actual file contents)
        self.raw_api = "https://raw.githubusercontent.com/"

        # GitHub contents API (for listing files)
        self.contents_api = "https://api.github.com/repos/"

    def _is_excluded(self, name: str):
        return any(name.endswith(ext) for ext in self.exclude_ext)

    def _get_extension(self, name: str):
        return os.path.splitext(name)[1] if "." in name else ""

    def _get_repo_name(self, repo_url: str):
        """
        Converts github repo URL into owner/repo format.
        Handles trailing slashes and .git suffix in any order.
        """
        if isinstance(repo_url, str):
            # Strip whitespace, trailing slashes, .git suffix, trailing slashes again
            cleaned = repo_url.strip().rstrip("/").removesuffix(".git").rstrip("/")
            url_parts = cleaned.split("https://github.com/")[-1].split("/")
            if len(url_parts) >= 2 and all(url_parts[:2]):
                return "/".join(url_parts[:2])
            raise ValueError("Repository URL must look like https://github.com/owner/repo")
        raise TypeError("Kindly provide a string as input.")


    def _fetch_recursive(self, owner, repo, path="", branch="main"):
        """
        Recursively fetch GitHub directory tree using the Contents API.
        Returns flat list of all file metadata.
        """
        url = f"{self.contents_api}{owner}/{repo}/contents/{path}?ref={branch}"
        response = self.s.get(url)

        if response.status_code == 403:
            raise PermissionError(
                "GitHub returned 403 Forbidden. "
                "You may have hit the API rate limit. "
                "Set GITHUB_TOKEN=<your_token> in .env for higher limits."
            )
        if response.status_code == 429:
            raise PermissionError(
                "GitHub rate limit exceeded (429). "
                "Set GITHUB_TOKEN=<your_token> in .env and retry."
            )
        if response.status_code != 200:
            raise ValueError(f"Error {response.status_code}: {response.text}")

        items = response.json()

        # GitHub returns either dict (file) or list (directory)
        if isinstance(items, dict) and items.get("type") == "file":
            return [items]

        files = []
        for item in items:
            if item["type"] == "dir":
                files.extend(self._fetch_recursive(owner, repo, item["path"], branch))
            elif item["type"] == "file":
                files.append(item)

        return files


    def _get_default_branch(self, owner: str, repo: str) -> str:
        response = self.s.get(f"{self.contents_api}{owner}/{repo}")
        if response.status_code == 404:
            raise ValueError(
                f"Repository '{owner}/{repo}' not found. "
                "Check the URL and make sure the repo is public (or GITHUB_TOKEN is set)."
            )
        response.raise_for_status()
        return response.json().get("default_branch", "main")

    def get_dir_tree(self, repo_url: str, branch: str | None = None, refresh: bool = False) -> dict:
        """
        Returns nested tree structure of repository using the Contents API.

        Parameters
        ----------
        repo_url: Full GitHub HTTPS URL, e.g. https://github.com/owner/repo
        branch:   Branch to analyse; None means auto-detect the default branch.
        refresh:  When True, bypass the on-disk cache and re-fetch.
        """
        repo_name = self._get_repo_name(repo_url)
        owner, repo = repo_name.split("/")
        branch = branch or self._get_default_branch(owner, repo)

        # Cache key includes branch so different branches stay separate
        cache_url = f"tree::{owner}/{repo}@{branch}"

        if not refresh:
            cached = cache_load_json(cache_url)
            if cached is not None:
                print(f"\n[cache] Loaded repo tree for {repo_name}@{branch} from disk.")
                return cached

        all_files = self._fetch_recursive(owner, repo, "", branch)
        metadata: dict = {}

        for f in all_files:
            path = f["path"]
            ext = self._get_extension(path)

            if self._is_excluded(path):
                continue

            meta = {
                "path": path,
                "type": "file",
                "ext": ext,
                "size_kb": round(f.get("size", 0) / 1024, 2),
                "url": f"{self.raw_api}{owner}/{repo}/{branch}/{path}"
            }

            # Build nested metadata tree
            parts = path.split("/")
            cursor = metadata
            for folder in parts[:-1]:
                folder_key = folder + "/"
                cursor = cursor.setdefault(folder_key, {})

            cursor[parts[-1]] = meta

        # Persist to cache
        cache_save_json(cache_url, metadata)

        print(f"\n\nRepository metadata tree created {len(all_files)} total items.")
        return metadata
