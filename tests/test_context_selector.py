import asyncio

from langchain_core.messages import HumanMessage

from evaluation.run import evaluate
from src.nodes.summarize_repo_node import summarize_repo_node
from src.retrieval.context_selector import (
    COMPACT_CONTEXT_BUDGET,
    COMPACT_MAX_EXCERPTS,
    select_compact_context,
)
from src.retrieval.evidence import Evidence


def test_compact_selector_is_ordered_bounded_and_deduplicates_same_file_spans():
    chunks = [
        Evidence("a.py", 1, 10, "def answer():\n    if ready:\n        return 1"),
        Evidence("a.py", 3, 7, "if ready:\n    return 1"),  # contained in first
        Evidence("b.py", 1, 2, "def distract():\n    return 2"),
        Evidence("a.py", 11, 18, "def helper():\n    return 3"),
        Evidence("c.py", 1, 2, "def other():\n    return 4"),
    ]
    selected = select_compact_context(chunks)
    assert selected == [chunks[0], chunks[2], chunks[3], chunks[4]]
    assert len(selected) <= COMPACT_MAX_EXCERPTS
    assert sum(len(chunk.render()) + 2 for chunk in selected) <= COMPACT_CONTEXT_BUDGET
    assert selected[0].start_line == 1 and selected[0].end_line == 10
    assert "if ready" in selected[0].text and "return 1" in selected[0].text


def test_compact_selector_skips_oversized_blocks_without_prefix_truncation():
    oversized = Evidence("a.py", 7, 9, "x" * (COMPACT_CONTEXT_BUDGET + 1))
    useful = Evidence("b.py", 4, 5, "def f():\n    return 1")
    selected = select_compact_context([oversized, useful])
    assert selected == [useful]
    assert useful.text == "def f():\n    return 1"


def test_code_compact_node_records_actual_selected_context():
    state = {
        "parsed_files": [
            {"path": "a.py", "raw": "def answer():\n    if ready:\n        return 1\n"},
            {"path": "b.py", "raw": "def distract():\n    return 2\n"},
        ],
        "messages": [HumanMessage(content="What does answer return?")],
        "llm": None,
        "retrieval_mode": "code_compact",
    }
    result = asyncio.run(summarize_repo_node(state))
    metrics = result["metrics"]
    assert metrics["context_selector"] == "compact_v1"
    assert metrics["context_budget_chars"] == COMPACT_CONTEXT_BUDGET
    assert metrics["candidate_chunks"] >= metrics["selected_chunks"]
    assert metrics["selected_chunks"] <= COMPACT_MAX_EXCERPTS
    assert metrics["selected_context_chars"] <= COMPACT_CONTEXT_BUDGET
    assert len(result["sources"]) == metrics["selected_chunks"]


def test_code_compact_benchmark_measures_selected_context_not_raw_k():
    files = [{"path": "a.py", "raw": "def answer():\n    return 1\n"},
             {"path": "b.py", "raw": "def distract():\n    return 2\n"}]
    questions = [{"id": "q1", "question": "What does answer return?",
                  "path": "a.py", "anchor": "return 1"}]
    result = evaluate(files, questions, mode="code_compact")
    row = result["rows"][0]
    assert result["config"]["context_selector"] == "compact_v1"
    assert row["candidate_chunks"] >= row["selected_chunks"]
    assert row["selected_chunks"] <= 4
    assert row["selected_context_chars"] <= 8000
    assert row["evidence_hit"] == 1
