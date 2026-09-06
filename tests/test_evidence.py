import asyncio
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from src.retrieval.evidence import build_chunks, retrieve, bounded_context
from src.retrieval.ranker import rank_chunks
from src.retrieval.chunker import chunk_text
from src.nodes.summarize_repo_node import summarize_repo_node


def test_source_lines_and_body_survive_chunking():
    raw = "import os\nasync def fetch():\n    return 42\n"
    chunks = build_chunks([{"path":"src/a.py", "raw":raw}], 2, 1)
    assert chunks[-1].start_line == 2
    assert chunks[-1].end_line == 3
    assert "return 42" in chunks[-1].render()
    assert "src/a.py:L2-L3" in chunks[-1].render()


def test_duplicate_content_keeps_distinct_files():
    files = [{"path":p, "raw":"def important(): pass"} for p in ("a.py", "b.py")]
    assert {c.path for c in retrieve("important", files)} == {"a.py", "b.py"}


def test_parsed_text_does_not_claim_source_line_numbers():
    chunk = build_chunks([{"path":"a.py", "parsed":"Function: hello"}])[0]
    assert "not source line numbers" in chunk.render()


def test_failed_fetch_is_not_evidence():
    assert build_chunks([{"path":"a.py", "parsed":"error", "error":True}]) == []


def test_context_budget_keeps_whole_chunks():
    chunks = build_chunks([{"path":"a.py", "raw":"one\ntwo\nthree"}], 1, 0)
    budget = len(chunks[0].render()) + 2
    assert bounded_context(chunks, budget) == [chunks[0]]


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, 10), (10, -1)])
def test_invalid_chunk_configuration_rejected(size, overlap):
    with pytest.raises(ValueError):
        build_chunks([], size, overlap)
    with pytest.raises(ValueError):
        chunk_text("text", size, overlap)


def test_ranks_even_when_k_exceeds_corpus_size():
    assert rank_chunks("needle", ["unrelated", "needle"], 8)[0] == "needle"


def test_zero_overlap_returns_no_evidence():
    assert rank_chunks("zzzzunique", ["unrelated content"], 8) == []


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_top_k_rejected(k):
    with pytest.raises(ValueError):
        rank_chunks("q", [], k)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        rank_chunks("q", ["q"], mode="typo")


def test_semantic_errors_do_not_silently_fall_back(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="Semantic retrieval unavailable"):
        rank_chunks("q", ["q"], mode="semantic")


def test_model_prompt_contains_cited_source_and_retrieval_mode(monkeypatch):
    import src.nodes.summarize_repo_node as module
    captured = {}
    original = module.retrieve
    def spy(query, files, top_k, mode):
        captured.update(top_k=top_k, mode=mode)
        return original(query, files, top_k, "lexical")
    monkeypatch.setattr(module, "retrieve", spy)
    llm = AsyncMock()
    llm.ainvoke.return_value = AIMessage(content="Returns 42 [a.py:L1-L2].")
    result = asyncio.run(summarize_repo_node({"messages":[HumanMessage(content="answer")],
        "parsed_files":[{"path":"a.py", "raw":"def answer():\n    return 42"}],
        "llm":llm, "top_k":1, "retrieval_mode":"semantic"}))
    assert captured == {"top_k":1, "mode":"semantic"}
    prompt = llm.ainvoke.call_args.args[0]
    assert "a.py:L1-L2" in prompt[1].content
    assert "return 42" in prompt[1].content
    assert "untrusted data" in prompt[0].content
    assert result["sources"][0]["path"] == "a.py"


def test_empty_evidence_does_not_call_model():
    llm = AsyncMock()
    result = asyncio.run(summarize_repo_node({"parsed_files":[], "llm":llm}))
    llm.ainvoke.assert_not_called()
    assert "cannot answer" in result["summary"]


def test_retrieval_only_identifies_itself():
    result = asyncio.run(summarize_repo_node({"messages":[HumanMessage(content="value")],
        "parsed_files":[{"path":"a.py", "raw":"value = 42"}], "llm":None}))
    assert "no LLM answer generated" in result["summary"]
    assert not result["metrics"]["generated"]
