"""
src/nodes/fetch_and_parse_node.py
===================================
Fetches file content for *selected_files* and parses each one using the
appropriate parser tool.

Concurrency
-----------
Files are fetched concurrently (up to ``MAX_CONCURRENT_FETCHES`` in
parallel) using ``asyncio.to_thread`` so the blocking ``requests`` calls
do not stall the event loop.  Output order is deterministic (same order
as *selected_files*).

Progress
--------
A ``tqdm`` progress bar is shown on stderr during fetching.  It is
automatically suppressed when stderr is not a TTY (e.g. in CI or when
output is piped) or when the ``TQDM_DISABLE`` environment variable is set
to ``"1"``.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from langchain_core.messages import SystemMessage

from src.tools.parse_python   import parse_python
from src.tools.parse_markdown  import parse_markdown
from src.tools.parse_notebook  import parse_notebook
from src.tools.parse_json_yaml import parse_json_yaml
from src.utils.fetch_blob      import fetch_blob_content
from src.utils.logger          import get_logger
from src.config.settings       import MAX_CONCURRENT_FETCHES

log = get_logger(__name__)

# ── Parser registry ──────────────────────────────────────────────────────────
PARSERS = {
    ".py":    parse_python,
    ".md":    parse_markdown,
    ".txt":   parse_markdown,
    ".json":  parse_json_yaml,
    ".yaml":  parse_json_yaml,
    ".yml":   parse_json_yaml,
    ".ipynb": parse_notebook,
}

# ── tqdm import (optional) ───────────────────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TQDM_AVAILABLE = False


def _tqdm_bar(iterable, total: int, desc: str):
    """Return a tqdm bar if available and stderr is a TTY, else a plain list."""
    disable = (
        not _TQDM_AVAILABLE
        or os.getenv("TQDM_DISABLE", "0") == "1"
        or not hasattr(__import__("sys").stderr, "isatty")
        or not __import__("sys").stderr.isatty()
    )
    if _TQDM_AVAILABLE and not disable:
        return _tqdm(iterable, total=total, desc=desc, unit="file", dynamic_ncols=True)
    return iterable


# ── Core helpers ─────────────────────────────────────────────────────────────

def _parse(raw: str, ext: str, path: str) -> str:
    """Run the appropriate parser; fall back to raw truncation on error."""
    parser_fn = PARSERS.get(ext)
    if parser_fn:
        try:
            result = parser_fn(raw)
            log.debug("Parsed %s with %s parser.", path, ext)
            return result
        except Exception as exc:  # noqa: BLE001
            log.warning("Parser error for %s: %s", path, exc)
            return f"<Error parsing {path}: {exc}>"
    # Unknown extension — safe truncation
    return raw[:5000]


async def _fetch_one(
    file_meta: dict,
    refresh: bool,
    semaphore: asyncio.Semaphore,
    progress_callback,
) -> Optional[dict]:
    """
    Fetch and parse a single file.

    Returns a parsed-file dict, or None if the file should be skipped
    (already parsed or fetch failed).
    """
    path = file_meta.get("path", "")
    url  = file_meta.get("url", "")
    ext  = file_meta.get("ext", "").lower()

    if not path or not url:
        progress_callback()
        return None

    async with semaphore:
        log.debug("Fetching %s", path)
        raw = await asyncio.to_thread(fetch_blob_content, url, refresh)

    progress_callback()

    if not raw:
        log.warning("Empty content for %s", path)
        return {"path": path, "ext": ext, "parsed": f"<Failed to fetch {path}>", "error": True}

    parsed = _parse(raw, ext, path)
    return {"path": path, "ext": ext, "parsed": parsed, "raw": raw}


# ── Node ─────────────────────────────────────────────────────────────────────

async def fetch_and_parse_node(state: dict) -> dict:
    """
    Fetch *selected_files* concurrently and parse each one.

    - Up to ``MAX_CONCURRENT_FETCHES`` requests run in parallel.
    - Files already present in ``parsed_files`` are skipped (cache hit).
    - A tqdm progress bar is shown when running interactively.
    - Output order mirrors the order of *selected_files*.
    """
    selected_files: list[dict] = state.get("selected_files", [])
    refresh: bool = bool(state.get("refresh_cache", False))

    if not selected_files:
        return {
            "parsed_files": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content="No files selected for parsing.")
            ],
        }

    # Skip already-parsed files
    selected_paths = {f.get("path") for f in selected_files}
    existing_pf = [] if refresh else [
        f for f in (state.get("parsed_files") or [])
        if f.get("path") in selected_paths and not f.get("error")
    ]
    already_parsed = {pf["path"] for pf in existing_pf if "path" in pf}

    to_fetch = [m for m in selected_files if m.get("path") not in already_parsed]

    if not to_fetch:
        log.info("All %d selected files already parsed (cache hit).", len(selected_files))
        return {
            "parsed_files": existing_pf,
            "messages": state.get("messages", []) + [
                SystemMessage(content="All selected files already parsed.")
            ],
        }

    log.info(
        "Fetching %d files concurrently (max=%d, %d already cached).",
        len(to_fetch), MAX_CONCURRENT_FETCHES, len(already_parsed),
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    # ── Progress bar ──────────────────────────────────────────────────────
    pbar = _tqdm_bar(range(len(to_fetch)), total=len(to_fetch), desc="Fetching files")
    pbar_iter = iter(pbar) if hasattr(pbar, "__iter__") else iter(range(len(to_fetch)))

    def _tick():
        try:
            next(pbar_iter)
        except StopIteration:
            pass
        if hasattr(pbar, "update"):
            pbar.update(1)

    # ── Launch all fetch tasks ────────────────────────────────────────────
    tasks = [
        _fetch_one(meta, refresh, semaphore, _tick)
        for meta in to_fetch
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    if hasattr(pbar, "close"):
        pbar.close()

    # ── Collect results (preserve order) ─────────────────────────────────
    new_pf: list[dict] = []
    for meta, result in zip(to_fetch, results):
        if isinstance(result, Exception):
            log.error("Unhandled error fetching %s: %s", meta.get("path"), result)
            new_pf.append({
                "path": meta.get("path", "<unknown>"),
                "ext":  meta.get("ext", ""),
                "parsed": f"<Fetch error: {result}>",
                "error": True,
            })
        elif result is not None:
            new_pf.append(result)

    all_pf = existing_pf + new_pf
    print(f"Fetched & parsed {len(new_pf)} files ({len(already_parsed)} from cache).")

    return {
        "parsed_files": all_pf,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Fetched & parsed {len(new_pf)} files.")
        ],
    }
