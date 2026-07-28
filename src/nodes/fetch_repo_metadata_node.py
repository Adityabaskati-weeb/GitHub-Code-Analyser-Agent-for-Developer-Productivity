# nodes/fetch_repo_metadata_node.py
from langchain_core.messages import SystemMessage
from src.github_repo_parser import GitRepoParser

async def fetch_repo_metadata_node(state: dict) -> dict:
    """
    First node of the workflow:
    - Reads repository URL from state['url']
    - Optionally reads state['branch'] and state['refresh_cache']
    - Calls GitRepoParser to get metadata tree (with disk caching)
    - Updates state with repo_tree
    """

    repo_url = state.get("url", None)
    if not repo_url:
        return {
            "messages": state["messages"] + [
                SystemMessage(content="No repository URL provided.")
            ]
        }

    branch = state.get("branch") or None
    refresh = bool(state.get("refresh_cache", False))

    try:
        parser = GitRepoParser()
        repo_tree = parser.get_dir_tree(repo_url, branch=branch, refresh=refresh)

        return {
            "repo_tree": repo_tree,
            "messages": state["messages"] + [
                SystemMessage(content=f"Fetched metadata tree for: {repo_url}")
            ]
        }

    except PermissionError as e:
        # GitHub rate limit or auth errors
        err = str(e)
        print(err)
        return {
            "repo_tree": {},
            "messages": state["messages"] + [
                SystemMessage(content=err)
            ]
        }

    except ValueError as e:
        # Bad URL or 404
        err = str(e)
        print(err)
        return {
            "repo_tree": {},
            "messages": state["messages"] + [
                SystemMessage(content=err)
            ]
        }

    except Exception as e:
        err = f"Error fetching repository metadata: {e}"
        print(err)
        return {
            "repo_tree": {},
            "messages": state["messages"] + [
                SystemMessage(content=err)
            ]
        }
