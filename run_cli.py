import argparse
import asyncio
import sys
from langchain_core.messages import HumanMessage

from langgraph_app import index_app, qa_app
from src.config.settings import get_llm
from state_schema import Agent_State

async def load_repo(repo_url: str) -> Agent_State:

    # Initial state for the graph
    state: Agent_State = {
        "messages": [],
        "url": repo_url,
        "repo_tree": {},
        "global_context": None,
        "selected_files": [],
        "unselected_files": [],
        "parsed_files": [],
        "intent": "",
        "keywords": [],
        "targets": {},
        "summary": [],
        "llm": get_llm(),
    }

    # print(f"\nStarting analysis for repo:\n{repo_url}\n")

    async for step in index_app.astream(state):
        # step is a dict: {"node_name": updated_state}
        # print(step)
        _, delta = list(step.items())[0]
        # print(state)
        # print(f"Indexing Node executed: {node_name}")
        state.update(delta)

        # Optionally print the node's output
        # updated_state = step[node_name]
        # if "messages" in updated_state:
        #     last_msg = updated_state["messages"][-1]
            # print(f"Message: {last_msg.content if hasattr(last_msg, 'content') else last_msg}")

    print("\nFinished Indexing Repository.\n")
    return state

async def qa(repo_state: Agent_State, question: str) -> Agent_State:
    state: Agent_State = {
        **repo_state,
        "messages": [HumanMessage(content=question)],
        "intent": "",
        "keywords": [],
        "targets": {},
        "selected_files": [],
        "unselected_files": [],
        "summary": "",
    }

    async for step in qa_app.astream(state):
        _, delta = list(step.items())[0]
        # print(f"QA Node excuted: {node_name}")
        state.update(delta)

    return state

async def qa_loop(repo_state: Agent_State):
    current_state = repo_state

    while True:
        try:
            user_query = input("\nYou (or 'exit' to quit): \n\n").strip()
        except(EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            print("\nGOODBYE...")
            break
        current_state = await qa(current_state, user_query)
        summary = current_state.get("summary") or "(No answer generated.)"
        print("\nAgent: \n")
        print(summary)

def smoke_test() -> int:
    from src.nodes.query_analyser_node import detect_intent, extract_keywords, extract_targets
    from src.tools.parse_python import parse_python
    from src.utils.flatten_tree import flatten_tree

    sample_tree = {
        "src/": {
            "app.py": {
                "path": "src/app.py",
                "type": "file",
                "ext": ".py",
                "size_kb": 1.0,
                "url": "https://example.com/src/app.py",
            }
        }
    }
    flattened = flatten_tree(sample_tree)
    parsed = parse_python("import os\n\ndef run_app():\n    return 'ok'\n")
    intent = detect_intent("explain the pipeline flow")
    keywords = extract_keywords("where is run_app used")
    targets = extract_targets("where is run_app() used")

    assert flattened and flattened[0]["path"] == "src/app.py"
    assert "run_app" in parsed
    assert intent == "pipeline_flow"
    assert "run_app" in keywords
    assert targets["function"] == "run_app"

    print("Smoke test passed.")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Index a GitHub repository and ask developer-productivity questions about the codebase."
    )
    parser.add_argument("repo_url", nargs="?", help="GitHub repository URL, for example https://github.com/owner/repo")
    parser.add_argument("-q", "--question", help="Ask one question and exit instead of opening the interactive loop.")
    parser.add_argument("--smoke-test", action="store_true", help="Run local parser/utility checks without network or LLM calls.")
    return parser.parse_args()


async def main():
    args = parse_args()

    if args.smoke_test:
        return smoke_test()

    if not args.repo_url:
        print("Usage: python run_cli.py <github_repo_url>")
        print("Try: python run_cli.py --smoke-test")
        return 1

    try:
        repo_url = args.repo_url
        repo_state = await load_repo(repo_url)
        print("Repository indexed. You can now ask questions about the codebase.")

        if args.question:
            answered_state = await qa(repo_state, args.question)
            print("\nAgent:\n")
            print(answered_state.get("summary") or "(No answer generated.)")
            return 0

        await qa_loop(repo_state)
        return 0
    except Exception as error:
        message = str(error)
        if "RESOURCE_EXHAUSTED" in message or "429" in message:
            print("Gemini quota exceeded. Try a lighter GOOGLE_MODEL or retry later.", file=sys.stderr)
        else:
            print(f"Error: {message}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))



#     if state.get("summary"):
#         print(f"FINAL SUMMARY:\n{state.get('summary')}")
#     else:
#         print("No summary generated.")

# if __name__ == "__main__":
#     # Expect repo URL as CLI argument
#     if len(sys.argv) < 2:
#         print("Usage: python run_cli.py <github_repo_url>")
#         sys.exit(1)

#     repo_url = sys.argv[1]

    # asyncio.run(run(repo_url))
