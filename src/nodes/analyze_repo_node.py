import re
from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.flatten_tree import flatten_tree
from src.config.settings import IMPORTANT_EXT, IMPORTANT_NAMES

def matches_keywords(path, keywords):
        path_lower = path.lower()
        return any(kw.lower() in path_lower for kw in keywords)

def add_unique(meta, selected_list, selected_paths: set):
    p = meta["path"].lower()
    if p not in selected_paths:
        selected_paths.add(p)
        selected_list.append(meta)

async def analyze_tree_node(state: dict) -> dict:
    """
    Query-aware Analyze Node.
    Decides which files to parse deeply based on:
      - repo metadata
      - user query (latest HumanMessage)
      - file importance (README, setup, main, config, etc.)
      - file sizes and extensions
    """

    # print("Initializing Analyze Tree Node...")

    repo_tree = state.get("repo_tree", None)
    intent = state.get("intent")
    keywords = state.get("keywords", [])
    targets = state.get("targets", {})
    
    if not repo_tree:
        return {
                "selected_files": [],
                "unselected_files": [],
                "messages": state.get("messages", []) + [
                    SystemMessage(content="No repository tree available.")
                ]}

    flattened = flatten_tree(repo_tree)

    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_query = msg.content.lower()
            break

    query_keywords = re.findall(r"[a-zA-Z_]+", user_query)

    selected = list(state.get("selected_files") or [])
    selected_paths = set()
    unselected = list(state.get("unselected_files") or [])

    for meta in flattened:
        path = meta["path"].lower()
        ext = meta["ext"].lower()

        selected_files = False

        # 1) Always include important names
        if any(name in path for name in IMPORTANT_NAMES):
            selected_files = True

        # 2) Match keywords from user query
        elif any(k in path for k in query_keywords):
            selected_files = True

        # 3) Keep primary code/docs
        elif ext in IMPORTANT_EXT:
            selected_files = True

        # 4) Intent-specific boosts
        if intent == "function_usage":
            fn = targets.get("function")
            if fn and fn.lower() in path:
                selected_files = True
            elif matches_keywords(path, keywords):
                selected_files = True

        elif intent == "type_lookup":
            var = targets.get("variable")
            if var and var.lower() in path:
                selected_files = True
            elif matches_keywords(path, keywords):
                selected_files = True

        elif intent == "directory_question":
            directory = targets.get("directory")
            if directory and path.startswith(directory):
                selected_files = True

        elif intent == "pipeline_flow":
            pipeline_markers = ["train", "main", "pipeline", "runner", "engine"]
            if any(m in path for m in pipeline_markers):
                selected_files = True

        elif intent == "architecture_summary":
            if path.count("/") <= 1 and path.endswith(".py"):
                selected_files = True

        elif intent == "high_level_summary":
            if matches_keywords(path, keywords):
                selected_files = True

        # Final push to lists (no duplicates)
        if selected_files:
            add_unique(meta, selected, selected_paths)
        else:
            unselected.append(meta)

    print(f"\nSelected {len(selected)} files, unselected {len(unselected)}.")

    return {
        "selected_files": selected,
        "unselected_files": unselected,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Selected {len(selected)} files based on query: '{user_query}'.")
        ]
    }
