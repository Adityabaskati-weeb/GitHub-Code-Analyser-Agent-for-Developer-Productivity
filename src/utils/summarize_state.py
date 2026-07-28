def summarize_state(state):
    out = {}
    for k, v in state.items():
        if isinstance(v, (list, tuple, set)):
            out[k] = f"<{type(v).__name__}> len={len(v)}"
        elif isinstance(v, dict):
            out[k] = f"<dict> keys={list(v.keys())[:5]} (total={len(v)})"
        else:
            out[k] = f"{type(v).__name__}: {str(v)[:80]}..."
    return out

#for debugging only
#Ouptut:
#{
#   "repo_tree": "<dict> keys=112, depth=4",
#   "selected_files": "<list> len=13",
#   "parsed_files": "<list> len=10",
#   "messages": "<list> len=2",
#   "intent": "function_usage",
#   "keywords": ["preprocess", "usage"],
#   "targets": {"function": "preprocess"},
#   ...
# }