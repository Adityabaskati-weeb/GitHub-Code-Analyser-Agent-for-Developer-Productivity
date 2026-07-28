"""
Script to parse Repository and store contents
in textual form.
"""
import os
import requests
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
        """
        if isinstance(repo_url, str):
            cleaned = repo_url.strip().removesuffix(".git").rstrip("/")
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
        response.raise_for_status()
        return response.json().get("default_branch", "main")

    def get_dir_tree(self, repo_url, branch=None):
        """
        Returns nested tree structure of repository using the Contents API.
        """
        repo_name = self._get_repo_name(repo_url)
        owner, repo = repo_name.split("/")
        branch = branch or self._get_default_branch(owner, repo)

        # print(f"Fetching directory tree for {repo_name}")

        all_files = self._fetch_recursive(owner, repo, "", branch)
        metadata = {}

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

        print(f"\n\nRepository metadata tree created {len(all_files)} total items.")
        # print(metadata)
        return metadata
