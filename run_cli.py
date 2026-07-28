"""
run_cli.py -- GitHub Code-Analyser Agent CLI
=============================================

Usage examples
--------------
# Local smoke-test (no network, no API key required)
python run_cli.py --smoke-test

# Index a repo and open the interactive Q&A loop
python run_cli.py https://github.com/owner/repo

# One-shot question (saves a Markdown report to reports/)
python run_cli.py https://github.com/owner/repo -q "Explain the architecture"

# Analyse a specific branch
python run_cli.py https://github.com/owner/repo --branch develop -q "What changed?"

# Force-refresh the cache
python run_cli.py https://github.com/owner/repo --refresh-cache -q "Show pipeline"

# Verbose debug output
python run_cli.py https://github.com/owner/repo -q "Explain X" --verbose

# Top-5 chunks, semantic ranking, suppress report
python run_cli.py https://github.com/owner/repo -q "Explain X" --top-k 5 \\
    --retrieval-mode semantic --no-report
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from langchain_core.messages import HumanMessage

from langgraph_app import index_app, qa_app
from state_schema import Agent_State


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_api_key() -> None:
    """Exit early with a helpful message if no Gemini key is configured."""
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        print(
            "Error: No Gemini API key found.\n"
            "  Set GOOGLE_API_KEY=<your_key> in your .env file or environment.\n"
            "  See .env.example for details.",
            file=sys.stderr,
        )
        sys.exit(1)


def _get_llm():
    from src.config.settings import get_llm
    return get_llm()


def _try_save_report(
    repo_url: str,
    question: str,
    state: Agent_State,
    branch: str | None,
    no_report: bool,
) -> None:
    """Save a Markdown report unless --no-report was given."""
    if no_report:
        return
    try:
        from src.report.writer import write_report
        report_path = write_report(
            repo_url=repo_url,
            question=question,
            selected_files=state.get("selected_files", []),
            answer=state.get("summary", ""),
            branch=branch,
        )
        print(f"\nReport saved -> {report_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not save report: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Core workflows
# ---------------------------------------------------------------------------

async def load_repo(
    repo_url: str,
    branch: str | None = None,
    refresh_cache: bool = False,
    top_k: int | None = None,
) -> Agent_State:
    """Index a GitHub repository and return the populated agent state."""
    state: Agent_State = {
        "messages": [],
        "url": repo_url,
        "branch": branch,
        "refresh_cache": refresh_cache,
        "repo_tree": {},
        "global_context": None,
        "selected_files": [],
        "unselected_files": [],
        "parsed_files": [],
        "intent": "",
        "keywords": [],
        "targets": {},
        "summary": [],
        "llm": _get_llm(),
    }

    if top_k is not None:
        state["top_k"] = top_k  # type: ignore[typeddict-unknown-key]

    async for step in index_app.astream(state):
        _, delta = list(step.items())[0]
        state.update(delta)

    print("\nFinished Indexing Repository.\n")
    return state


async def qa(
    repo_state: Agent_State,
    question: str,
    top_k: int | None = None,
) -> Agent_State:
    """Run one Q&A turn and return the updated state."""
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

    if top_k is not None:
        state["top_k"] = top_k  # type: ignore[typeddict-unknown-key]

    async for step in qa_app.astream(state):
        _, delta = list(step.items())[0]
        state.update(delta)

    return state


async def qa_loop(repo_state: Agent_State, args: argparse.Namespace) -> None:
    """Interactive multi-turn Q&A loop with optional per-turn report saving."""
    current_state = repo_state

    while True:
        try:
            user_query = input("\nYou (or 'exit' to quit): \n\n").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            print("\nGOODBYE...")
            break

        current_state = await qa(current_state, user_query, top_k=args.top_k)
        summary = current_state.get("summary") or "(No answer generated.)"
        print("\nAgent:\n")
        print(summary)

        # Save a report for each interactive answer unless suppressed
        _try_save_report(
            repo_url=args.repo_url,
            question=user_query,
            state=current_state,
            branch=args.branch,
            no_report=args.no_report,
        )


# ---------------------------------------------------------------------------
# Smoke test (no network, no LLM, no API key)
# ---------------------------------------------------------------------------

def smoke_test() -> int:
    """
    Fast local sanity-check.
    Exercises: flatten_tree, parse_python, detect_intent,
               extract_keywords, extract_targets, chunk_text, rank_chunks,
               _is_skippable, _score.

    Returns 0 on success, 1 on failure.
    """
    from src.nodes.query_analyser_node import detect_intent, extract_keywords, extract_targets
    from src.tools.parse_python import parse_python
    from src.utils.flatten_tree import flatten_tree
    from src.retrieval.chunker import chunk_text
    from src.retrieval.ranker import rank_chunks
    from src.nodes.analyze_repo_node import _is_skippable, _score

    errors: list[str] = []

    # --- flatten_tree ---
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
    if not (flattened and flattened[0]["path"] == "src/app.py"):
        errors.append("flatten_tree: unexpected result")

    # --- parse_python ---
    parsed = parse_python("import os\n\ndef run_app():\n    return 'ok'\n")
    if "run_app" not in parsed:
        errors.append("parse_python: function name not found in output")

    # --- detect_intent ---
    intent = detect_intent("explain the pipeline flow")
    if intent != "pipeline_flow":
        errors.append(f"detect_intent: expected 'pipeline_flow', got '{intent}'")

    # --- extract_keywords ---
    keywords = extract_keywords("where is run_app used")
    if "run_app" not in keywords:
        errors.append("extract_keywords: 'run_app' not in keywords")

    # --- extract_targets ---
    targets = extract_targets("where is run_app() used")
    if targets.get("function") != "run_app":
        errors.append(f"extract_targets: expected function='run_app', got {targets}")

    # --- chunk_text ---
    long_text = " ".join([f"word{i}" for i in range(1000)])
    chunks = chunk_text(long_text, chunk_size=50, overlap=10)
    if len(chunks) < 2:
        errors.append(f"chunk_text: expected multiple chunks, got {len(chunks)}")

    # --- rank_chunks ---
    sample_chunks = ["the pipeline starts here", "import os and sys", "def run_app(): pass"]
    ranked = rank_chunks("pipeline flow", sample_chunks, top_k=2, mode="lexical")
    if not ranked:
        errors.append("rank_chunks: returned empty list")
    if len(ranked) > 2:
        errors.append(f"rank_chunks: expected <=2 chunks, got {len(ranked)}")

    # --- _is_skippable ---
    if not _is_skippable("vendor/jquery.min.js", ".js"):
        errors.append("_is_skippable: should skip *.min.js")
    if not _is_skippable("styles/main.min.css", ".css"):
        errors.append("_is_skippable: should skip *.min.css")
    if not _is_skippable("dist/bundle.js.map", ".map"):
        errors.append("_is_skippable: should skip *.js.map")
    if _is_skippable("src/main.py", ".py"):
        errors.append("_is_skippable: should NOT skip .py files")

    # --- _score (basic sanity) ---
    readme_meta = {"path": "README.md", "ext": ".md", "size_kb": 5}
    deep_meta   = {"path": "a/b/c/d/e/helper.py", "ext": ".py", "size_kb": 5}
    score_readme = _score(readme_meta, ["readme"], "high_level_summary", [], {})
    score_deep   = _score(deep_meta,   ["readme"], "high_level_summary", [], {})
    if score_readme <= score_deep:
        errors.append("_score: README should outscore deeply nested helper")

    if errors:
        for err in errors:
            print(f"FAIL: {err}", file=sys.stderr)
        return 1

    print("Smoke test passed - all checks OK")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Index a GitHub repository and ask developer-productivity "
            "questions about the codebase."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cli.py --smoke-test
  python run_cli.py https://github.com/owner/repo
  python run_cli.py https://github.com/owner/repo -q "Explain the architecture"
  python run_cli.py https://github.com/owner/repo --branch develop -q "What changed?"
  python run_cli.py https://github.com/owner/repo --refresh-cache -q "Show pipeline"
  python run_cli.py https://github.com/owner/repo -q "Explain X" --top-k 5 --no-report
  python run_cli.py https://github.com/owner/repo -q "Debug" --verbose
""",
    )
    parser.add_argument(
        "repo_url",
        nargs="?",
        help="GitHub repository URL, e.g. https://github.com/owner/repo",
    )
    parser.add_argument(
        "-q", "--question",
        help="Ask one question and exit (one-shot mode). Saves a Markdown report unless --no-report.",
    )
    parser.add_argument(
        "--branch",
        metavar="BRANCH",
        default=None,
        help="Git branch to analyse (default: repo default branch).",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Bypass the on-disk cache and re-fetch all GitHub data.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="N",
        help="Number of top-ranked file chunks sent to the LLM (default: 8).",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["lexical", "semantic"],
        default=None,
        help="Chunk ranking strategy (default: lexical).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Suppress Markdown report export.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for troubleshooting.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run local parser/utility checks without network or LLM calls.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    args = parse_args()

    # ---- Verbose logging ----
    if args.verbose:
        from src.utils.logger import set_verbose
        set_verbose(True)

    # ---- Smoke test (no API key needed) ----
    if args.smoke_test:
        return smoke_test()

    # ---- Validate repo URL ----
    if not args.repo_url:
        print(
            "Usage: python run_cli.py <github_repo_url> [options]\n"
            "Try:   python run_cli.py --smoke-test\n"
            "Help:  python run_cli.py --help",
            file=sys.stderr,
        )
        return 1

    # ---- Check API key early ----
    _check_api_key()

    # ---- Apply retrieval-mode override ----
    if args.retrieval_mode:
        os.environ["RETRIEVAL_MODE"] = args.retrieval_mode
        import src.config.settings as _s
        _s.RETRIEVAL_MODE = args.retrieval_mode

    # ---- Validate URL format before hitting the network ----
    repo_url = args.repo_url.strip()
    if "github.com/" not in repo_url:
        print(
            f"Error: '{repo_url}' doesn't look like a GitHub URL.\n"
            "  Expected format: https://github.com/owner/repo",
            file=sys.stderr,
        )
        return 1

    try:
        repo_state = await load_repo(
            repo_url,
            branch=args.branch,
            refresh_cache=args.refresh_cache,
            top_k=args.top_k,
        )
        print("Repository indexed. You can now ask questions about the codebase.")

        if args.question:
            answered_state = await qa(repo_state, args.question, top_k=args.top_k)
            answer = answered_state.get("summary") or "(No answer generated.)"
            print("\nAgent:\n")
            print(answer)

            _try_save_report(
                repo_url=repo_url,
                question=args.question,
                state=answered_state,
                branch=args.branch,
                no_report=args.no_report,
            )
            return 0

        await qa_loop(repo_state, args)
        return 0

    except PermissionError as e:
        print(f"GitHub Error: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        msg = str(e)
        if "must look like" in msg or "not found" in msg:
            print(f"Invalid repository: {msg}", file=sys.stderr)
        else:
            print(f"Error: {msg}", file=sys.stderr)
        return 1

    except Exception as error:
        message = str(error)
        if "RESOURCE_EXHAUSTED" in message or "429" in message:
            print(
                "Gemini quota exceeded. "
                "Try a lighter model (GOOGLE_MODEL=gemini-1.5-flash) or wait and retry.",
                file=sys.stderr,
            )
        elif "GOOGLE_API_KEY" in message or "GEMINI_API_KEY" in message:
            print(
                "Error: Missing or invalid Gemini API key.\n"
                "  Set GOOGLE_API_KEY=<your_key> in your .env file.",
                file=sys.stderr,
            )
        else:
            print(f"Error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
