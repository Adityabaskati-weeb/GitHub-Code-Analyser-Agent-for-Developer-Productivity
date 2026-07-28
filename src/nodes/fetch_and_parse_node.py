from langchain_core.messages import SystemMessage
from src.tools.parse_python import parse_python
from src.tools.parse_markdown import parse_markdown
from src.tools.parse_notebook import parse_notebook
from src.tools.parse_json_yaml import parse_json_yaml
from src.utils.fetch_blob import fetch_blob_content



# Tool registry
PARSERS = {
    ".py": parse_python,
    ".md": parse_markdown,
    ".txt": parse_markdown,
    ".json": parse_json_yaml,
    ".yaml": parse_json_yaml,
    ".yml": parse_json_yaml,
    ".ipynb": parse_notebook,
}

async def fetch_and_parse_node(state: dict) -> dict:
    """
    Fetches file content for 'selected_files' and parses each file using
    the appropriate tool for its extension.
    """

    # print("Initializing Fetch & Parse Node...")

    selected_files = state.get("selected_files", [])
    
    if not selected_files:
        return {
            "parsed_files": [],
            "messages": state.get("messages", []) + [
                SystemMessage(content="No files selected for parsing.")
            ]
        }
    
    parsed_files = list(state.get("parsed_files") or [])
    parsed_paths = {pf["path"] for pf in parsed_files if "path" in pf}
    new_pf = []
    
    for file_meta in selected_files:
        path = file_meta["path"]
        ext = file_meta["ext"].lower()
        url = file_meta["url"]

        print(f"Fetching file: {path}")

        if not path or not url:
            continue
        if path in parsed_paths:
            continue

        print(f"Fetching File: {path}")

        raw_content = fetch_blob_content(url)
        if not raw_content:
            new_pf.append({
                "path": path,
                "parsed": f"<Failed to fetch content for {path}>"
            })
            continue

        # Select parser based on extension
        parser_fn = PARSERS.get(ext, None)

        if parser_fn:
            try:
                print("Checking Execution of parser tools")
                parsed = parser_fn(raw_content)
                print(f"{path} file parsed using {ext} parser utility function")
            except Exception as e:
                parsed = f"<Error parsing file {path}: {e}>"
        else:
            parsed = raw_content[:5000]  # token-safe limit

        new_pf.append({
            "path": path,
            "ext": ext,
            "parsed": parsed
        })
        
    updated_pf = new_pf + parsed_files
    print(f"Total parsed files: {len(updated_pf)} files.")

    return {
        "parsed_files": parsed_files + new_pf,
        "messages": state.get("messages", []) + [
            SystemMessage(content=f"Fetched & parsed {len(new_pf)} files.")
        ]
    }
