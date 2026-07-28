# nodes/fetch_repo_metadata_node.py
from langchain_core.messages import SystemMessage
from src.github_repo_parser import GitRepoParser

async def fetch_repo_metadata_node(state: dict) -> dict:
    """
    First node of the workflow:
    - Reads repository URL from state['url']
    - Calls GitRepoParser to get metadata tree
    - Updates state with repo_tree
    """

    # print("Initializing Fetch Repo Metadata Node...")

    repo_url = state.get("url", None)
    if not repo_url:
        return {
            "messages": state["messages"] + [
                SystemMessage(content="No repository URL provided.")
            ]
        }

    try:
        parser = GitRepoParser()
        repo_tree = parser.get_dir_tree(repo_url)

        # print("Repo metadata tree fetched successfully!")
        # print(f"State Variable: {state}")
        return {
            "repo_tree": repo_tree,
            "messages": state["messages"] + [
                SystemMessage(content=f"Fetched metadata tree for: {repo_url}")
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
