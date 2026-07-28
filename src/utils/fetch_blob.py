"""
src/utils/fetch_blob.py
=======================
Fetches raw file content from a URL (typically a GitHub raw URL).
Results are transparently cached on disk to avoid redundant network
round-trips on repeated runs.
"""

from __future__ import annotations

import requests

from src.cache.disk_cache import load as cache_load, save as cache_save


def fetch_blob_content(blob_url: str, refresh: bool = False) -> str:
    """
    Fetch the text content at *blob_url*.

    Parameters
    ----------
    blob_url: The raw GitHub URL to download.
    refresh:  When ``True``, bypass the on-disk cache and re-download.

    Returns
    -------
    Decoded text (``str``), or an empty string on failure.
    """
    if not refresh:
        cached = cache_load(blob_url)
        if cached is not None:
            return cached

    try:
        response = requests.get(blob_url, timeout=30)
        response.raise_for_status()
        data = response.text
        cache_save(blob_url, data)
        return data

    except Exception as e:  # noqa: BLE001
        print(f"Error fetching blob content from {blob_url}: {e}")
        return ""
