"""
src/nodes/global_context_node.py
==================================
Builds a high-level natural-language summary of the repository structure.

File snippet fetches are routed through the cached :func:`fetch_blob_content`
utility (backed by the on-disk cache under ``.cache/``) so repeated runs do
not hit the GitHub network.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.flatten_tree import flatten_tree
from src.utils.fetch_blob import fetch_blob_content
from src.utils.logger import get_logger

log = get_logger(__name__)


async def global_context_node(state: dict) -> dict:
    """
    Builds a global overview of the repository based on the metadata tree
    fetched from GitRepoParser.

    Produces a brief summary of repo structure, key folders and relationships.
    File snippet fetches use the shared disk cache (refresh honoured via state).
    """
    repo_tree = state.get("repo_tree")
    refresh   = bool(state.get("refresh_cache", False))

    if not repo_tree:
        log.warning("No repo tree found in state; skipping global context.")
        return {"global_context": "No repo structure available"}

    flattened = flatten_tree(repo_tree)

    # Select up to 5 key files for snippet extraction
    imp_files = [
        f for f in flattened
        if any(
            kw in f["path"].lower()
            for kw in ["readme", "setup", "main", "app", "requirements", "scripts", "configs"]
        )
    ][:5]

    headers: list[str] = []
    for file_meta in imp_files:
        url  = file_meta.get("url", "")
        path = file_meta["path"]
        if not url:
            continue
        try:
            # Uses the cached fetch — no redundant GitHub round-trips
            raw     = fetch_blob_content(url, refresh=refresh)
            snippet = "\n".join(raw.splitlines()[:10])
            headers.append(f"{path}:\n{snippet}\n")
            log.debug("snippet fetched (cached): %s", path)
        except Exception as exc:  # noqa: BLE001
            headers.append(f"{path}: <Error fetching snippet: {exc}>")
            log.warning("Failed to fetch snippet for %s: %s", path, exc)

    tree_summ = "\n".join(
        f"- {f['path']} ({f['ext']}, {f['size_kb']} KB)"
        for f in flattened[:60]
    )

    prompt = f"""
You are an expert software architect.
Below is a summary of a GitHub repository structure and small snippets from key files.

### File Structure (first 60 files):
{tree_summ}

### Key File Headers:
{headers if headers else 'No key files found.'}

Please describe in 5-8 sentences:
1. The overall purpose of this repository.
2. The main components or modules and their likely roles.
3. How these modules might interact logically (e.g., data -> model -> evaluation).
4. Which parts appear to be core, supporting, or documentation.
"""

    system_msg = SystemMessage(content=(
        "You are an expert GitHub repository summariser. "
        "Provide insights on what functions and modules are present and how they "
        "connect to each other, in plain terms where possible."
    ))
    human_msg = HumanMessage(content=prompt)

    llm = state.get("llm")
    response = await llm.ainvoke([system_msg, human_msg])
    global_summ = response.content.strip()

    log.info("Global context summary generated (%d chars).", len(global_summ))
    return {"global_context": global_summ}
