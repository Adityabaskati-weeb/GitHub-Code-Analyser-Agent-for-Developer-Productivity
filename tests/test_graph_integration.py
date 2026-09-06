import asyncio
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage
import pytest

from run_cli import load_repo, qa
from src.github_repo_parser import GitRepoParser
from src.nodes.fetch_and_parse_node import fetch_and_parse_node


def test_graph_preserves_top_k_and_calls_model_only_for_question(monkeypatch):
    tree = {name:{"type":"file", "path":name, "ext":".py", "size_kb":1,
                  "url":f"https://example.com/{name}"} for name in ("first.py", "second.py")}
    monkeypatch.setattr(GitRepoParser, "get_dir_tree", lambda *a, **k: tree)
    monkeypatch.setattr("src.nodes.fetch_and_parse_node.fetch_blob_content",
                        lambda *a, **k: "def answer():\n    return 42\n")
    llm = AsyncMock()
    llm.ainvoke.return_value = AIMessage(content="42 [first.py:L1-L2]")
    monkeypatch.setattr("src.config.settings.get_llm", lambda *a, **k: llm)
    async def run():
        state = await load_repo("https://github.com/owner/repo", top_k=1)
        llm.ainvoke.assert_not_called()
        state["retrieval_mode"] = "lexical"
        result = await qa(state, "answer")
        assert result["metrics"]["retrieved_chunks"] == 1
        assert len(result["sources"]) == 1
        llm.ainvoke.assert_awaited_once()
    asyncio.run(run())


def test_metadata_failure_is_not_reported_as_success(monkeypatch):
    def fail(*a, **k):
        raise PermissionError("rate limit")
    monkeypatch.setattr(GitRepoParser, "get_dir_tree", fail)
    with pytest.raises(RuntimeError, match="No repository files indexed"):
        asyncio.run(load_repo("https://github.com/owner/repo", retrieval_only=True))


def test_refresh_refetches_parsed_content_and_drops_unselected_files(monkeypatch):
    calls = []
    def fetch(url, refresh):
        calls.append(refresh)
        return "value = 2"
    monkeypatch.setattr("src.nodes.fetch_and_parse_node.fetch_blob_content", fetch)
    result = asyncio.run(fetch_and_parse_node({
        "selected_files":[{"path":"a.py", "url":"https://example.com/a.py", "ext":".py"}],
        "parsed_files":[{"path":"a.py", "raw":"value = 1"}, {"path":"stale.py", "raw":"old"}],
        "refresh_cache":True, "messages":[]}))
    assert calls == [True]
    assert len(result["parsed_files"]) == 1
    assert result["parsed_files"][0]["raw"] == "value = 2"


@pytest.mark.parametrize("url", ["http://github.com/a/b", "https://evil.com/a/b",
    "https://github.com/a/b/tree/main", "https://github.com/a/b?token=secret"])
def test_parser_rejects_noncanonical_repo_urls(url):
    with pytest.raises(ValueError):
        GitRepoParser(github_token=False)._get_repo_name(url)
