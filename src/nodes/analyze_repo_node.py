"""
src/nodes/analyze_repo_node.py
================================
Query-aware file selection node with smart ranking and hard caps.

Selection pipeline
------------------
1. Skip minified files (*.min.js, *.min.css, *.map, …)
2. Skip files that exceed MAX_SIZE_KB
3. Score every remaining file by relevance to the query
4. Sort by score descending, keep top MAX_SELECTED_FILES
"""

from __future__ import annotations

import re
from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.flatten_tree import flatten_tree
from src.utils.logger import get_logger
from src.config.settings import (
    IMPORTANT_EXT,
    IMPORTANT_NAMES,
    MAX_SELECTED_FILES,
    MAX_SIZE_KB,
    SKIP_PATTERNS,
)

log = get_logger(__name__)

# ── Static/binary extensions that carry no semantic value ──────────────────
_SKIP_EXTENSIONS = {
    ".css", ".js",        # only if not caught by SKIP_PATTERNS first
    ".html", ".htm",      # kept out by default unless query-matched
    ".svg", ".ico", ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    ".zip", ".tar", ".gz", ".rar",
    ".lock",              # package-lock.json, yarn.lock etc.
}

# Extensions whose content is worth parsing (override _SKIP_EXTENSIONS)
_KEEP_EXTENSIONS = set(IMPORTANT_EXT)


def _is_skippable(path: str, ext: str) -> bool:
    """Return True if this file should never be selected."""
    lp = path.lower()
    # 1. Minified / source-map patterns
    if any(lp.endswith(pat) for pat in SKIP_PATTERNS):
        return True
    # 2. Binary / static-asset extensions that aren't in KEEP list
    if ext in _SKIP_EXTENSIONS and ext not in _KEEP_EXTENSIONS:
        return True
    return False


def _score(meta: dict, query_keywords: list[str], intent: str,
           keywords: list[str], targets: dict) -> int:
    """
    Return a relevance score for *meta*.
    Higher = more relevant = selected first when capping.
    """
    path = meta["path"].lower()
    ext  = meta["ext"].lower()
    score = 0

    # Tier 1 — always-important names (README, main, config, …)
    if any(name in path for name in IMPORTANT_NAMES):
        score += 40

    # Tier 2 — query keyword matches in path
    kw_hits = sum(1 for k in query_keywords if k in path)
    score += kw_hits * 20

    # Tier 3 — intent-specific boosts
    if intent == "function_usage":
        fn = targets.get("function", "")
        if fn and fn.lower() in path:
            score += 30
        if any(kw.lower() in path for kw in keywords):
            score += 15

    elif intent == "type_lookup":
        var = targets.get("variable", "")
        if var and var.lower() in path:
            score += 30
        if any(kw.lower() in path for kw in keywords):
            score += 15

    elif intent == "directory_question":
        directory = targets.get("directory", "")
        if directory and path.startswith(directory.lower()):
            score += 35

    elif intent == "pipeline_flow":
        pipeline_markers = ["train", "main", "pipeline", "runner", "engine"]
        if any(m in path for m in pipeline_markers):
            score += 25

    elif intent == "architecture_summary":
        if path.count("/") <= 1 and ext == ".py":
            score += 20

    # Tier 4 — important extension but no keyword match
    if ext in _KEEP_EXTENSIONS:
        score += 5

    # Penalise deeply nested paths (less likely to be core)
    depth = path.count("/")
    score -= depth * 2

    return score


async def analyze_tree_node(state: dict) -> dict:
    """
    Query-aware Analyze Node.

    Decides which files to parse deeply based on:
      - user query intent, keywords, targets
      - file importance (README, setup, main, config, …)
      - file extension (skip minified, binary, static assets)
      - file size   (skip files > MAX_SIZE_KB)
    Caps selection at MAX_SELECTED_FILES using a relevance score.
    """
    repo_tree = state.get("repo_tree")
    intent    = state.get("intent", "")
    keywords  = state.get("keywords", [])
    targets   = state.get("targets", {})

    if not repo_tree:
        return {
            "selected_files": [],
            "unselected_files": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content="No repository tree available.")
            ],
        }

    flattened = flatten_tree(repo_tree)

    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_query = msg.content.lower()
            break

    query_keywords = re.findall(r"[a-zA-Z_]+", user_query)

    # ── Pass 1: hard filters ─────────────────────────────────────────────
    candidates: list[dict] = []
    hard_skipped: list[str] = []

    for meta in flattened:
        path = meta["path"]
        ext  = meta.get("ext", "").lower()
        size = meta.get("size_kb", 0)

        # Skip minified / static-asset files
        if _is_skippable(path, ext):
            hard_skipped.append(f"{path} [minified/asset]")
            log.debug("SKIP (asset)   %s", path)
            continue

        # Skip oversized files
        if size > MAX_SIZE_KB:
            hard_skipped.append(f"{path} [size {size} KB > {MAX_SIZE_KB} KB]")
            log.debug("SKIP (size)    %s  %.1f KB", path, size)
            continue

        candidates.append(meta)

    # ── Pass 2: score and rank ────────────────────────────────────────────
    scored = [
        (_score(m, query_keywords, intent, keywords, targets), m)
        for m in candidates
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    # ── Pass 3: apply cap ────────────────────────────────────────────────
    selected_scored = scored[:MAX_SELECTED_FILES]
    overflow        = scored[MAX_SELECTED_FILES:]

    selected   = [m for _, m in selected_scored]
    unselected = (
        hard_skipped                          # already strings
        + [m["path"] for _, m in overflow]   # paths of capped files
    )

    log.info(
        "File selection: %d candidates, %d selected (cap=%d), %d hard-skipped, %d capped",
        len(candidates), len(selected), MAX_SELECTED_FILES,
        len(hard_skipped), len(overflow),
    )
    print(
        f"Selected {len(selected)}/{len(flattened)} files "
        f"(cap={MAX_SELECTED_FILES}, skipped {len(hard_skipped)} asset/oversized)."
    )

    return {
        "selected_files": selected,
        "unselected_files": unselected,
        "messages": state.get("messages", []) + [
            SystemMessage(
                content=(
                    f"Selected {len(selected)} files based on query "
                    f"(capped at {MAX_SELECTED_FILES})."
                )
            )
        ],
    }
