import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import evaluation.answers as answers
from evaluation.run import load_dataset
from src.nodes.summarize_repo_node import AnswerGenerationError, summarize_repo_node


def _setup_single_question(monkeypatch, llm=None):
    llm = llm or SimpleNamespace(model="test-model", timeout=1, retries=0)
    monkeypatch.setattr(answers, "get_llm", lambda *args: llm)
    monkeypatch.setattr(answers, "fingerprint", lambda: "dataset")
    monkeypatch.setattr(answers, "load_dataset", lambda: (
        {"files": [{"path": "a.py", "raw": "def f():\n    return 1\n"}], "revision": "r1"},
        [{"id": "q01", "question": "What does f return?", "answer": "1"}],
    ))
    return llm


def test_answer_outcome_and_real_total_time_are_persisted(tmp_path, monkeypatch):
    _setup_single_question(monkeypatch)
    monkeypatch.setattr(answers, "summarize_repo_node", AsyncMock(return_value={
        "summary": "Returns 1.", "sources": [], "model_output": "{}",
        "metrics": {"generated": True, "abstained": False,
                     "retrieved_chunks": 1, "inference_attempts": 1,
                     "retrieval_ms": 2.0, "generation_ms": 4.0,
                     "total_call_ms": 6.0},
    }))
    output = tmp_path / "answers.json"
    asyncio.run(answers.generate("test-model", output, question_ids=["q01"]))
    report = json.loads(output.read_text(encoding="utf-8"))
    row = report["rows"][0]
    assert row["outcome"] == row["metrics"]["outcome"] == "answered"
    assert row["metrics"]["total_call_ms"] == 6.0
    assert report["totals"]["logical_requests"] == 1
    assert report["totals"]["generation_calls"] == 1
    assert report["totals"]["inference_attempts"] == 1
    assert report["totals"]["total_call_ms"] == 6.0


def test_failure_checkpoint_keeps_sources_timings_and_attempts(tmp_path, monkeypatch):
    _setup_single_question(monkeypatch)
    error = RuntimeError("malformed structured answer")
    error._qa_sources = [{"path": "a.py", "start_line": 1, "end_line": 2,
                          "text": "def f():\n    return 1", "raw_source": True}]
    error._qa_context_chars = 42
    error._qa_retrieved_chunks = 1
    error._qa_retrieval_ms = 2.5
    error._qa_generation_ms = 7.5
    error._qa_total_call_ms = 10.0
    error._qa_model_output = '{"answer":'
    error._qa_inference = {"inference_attempts": 2, "termination_reason": "stop"}
    monkeypatch.setattr(answers, "summarize_repo_node", AsyncMock(side_effect=error))
    output = tmp_path / "answers.json"
    asyncio.run(answers.generate("test-model", output, question_ids=["q01"]))
    row = json.loads(output.read_text(encoding="utf-8"))["rows"][0]
    assert row["status"] == "failed"
    assert row["outcome"] == "malformed_response"
    assert row["sources"][0]["path"] == "a.py"
    assert row["model_output"] == '{"answer":'
    assert row["metrics"]["retrieval_ms"] == 2.5
    assert row["metrics"]["generation_ms"] == 7.5
    assert row["metrics"]["total_call_ms"] == 10.0
    assert row["metrics"]["inference_attempts"] == 2
    assert json.loads(output.read_text(encoding="utf-8"))["totals"]["failed_requests"] == 1


@pytest.mark.parametrize("metadata", [
    {"done": False}, {"termination_reason": "length"},
    {"termination_reason": "max_tokens"}, {"termination_reason": "limit"},
    {"termination_reason": "incomplete"},
])
def test_incomplete_response_is_rejected_before_rendering(metadata):
    llm = AsyncMock()
    llm.supports_source_schema = True
    llm.ainvoke.return_value = AIMessage(
        content=json.dumps({"answer": "Returns 1.", "source_ids": ["S1"],
                            "insufficient_evidence": False}),
        response_metadata=metadata,
    )
    state = {
        "parsed_files": [{"path": "a.py", "raw": "def f():\n    return 1\n"}],
        "messages": [HumanMessage(content="f")], "llm": llm,
    }
    with pytest.raises(AnswerGenerationError) as raised:
        asyncio.run(summarize_repo_node(state))
    error = raised.value
    assert error._qa_model_output
    assert error._qa_sources
    assert error._qa_total_call_ms >= error._qa_retrieval_ms


def test_pilot_selection_is_frozen_and_ordered():
    _, questions = load_dataset()
    selected = answers.select_questions(questions, pilot=True)
    assert [q["id"] for q in selected] == [
        "q03", "q07", "q11", "q15", "q19", "q23",
        "q28", "q32", "q36", "q40", "q44", "q48",
    ]


def test_resume_rejects_changed_configuration(tmp_path, monkeypatch):
    llm = SimpleNamespace(model="test-model", timeout=1, retries=0,
                          aget_run_metadata=AsyncMock(return_value={
                              "model_digest": "digest-1", "runtime_version": "runtime-1",
                          }))
    _setup_single_question(monkeypatch, llm)
    monkeypatch.setattr(answers, "summarize_repo_node", AsyncMock(return_value={
        "summary": "Returns 1.", "sources": [], "metrics": {
            "generated": True, "abstained": False, "retrieved_chunks": 1,
            "inference_attempts": 1, "retrieval_ms": 1.0,
            "generation_ms": 2.0, "total_call_ms": 3.0,
        },
    }))
    output = tmp_path / "answers.json"
    asyncio.run(answers.generate("test-model", output, question_ids=["q01"]))
    llm.aget_run_metadata.return_value = {"model_digest": "digest-2", "runtime_version": "runtime-1"}
    with pytest.raises(ValueError, match="different run configuration"):
        asyncio.run(answers.generate("test-model", output, resume=True,
                                     question_ids=["q01"]))
