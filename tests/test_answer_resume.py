import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import evaluation.answers as module


def setup_run(monkeypatch):
    monkeypatch.setattr(module, "get_llm", lambda *args: SimpleNamespace(model="test-model"))
    monkeypatch.setattr(module, "load_dataset", lambda: ({"files":[]}, [
        {"id":"q01", "question":"first", "answer":"one"},
        {"id":"q02", "question":"second", "answer":"two"}]))
    generate = AsyncMock(return_value={"summary":"answer", "sources":[], "metrics":{}})
    monkeypatch.setattr(module, "summarize_repo_node", generate)
    return generate


def test_resume_skips_completed_questions(tmp_path, monkeypatch):
    generate = setup_run(monkeypatch)
    output = tmp_path / "answers.json"
    output.write_text(json.dumps({"model":"test-model", "dataset_sha256":module.fingerprint(),
        "reviewer_type":"unassigned", "answer_protocol":"source_ids_v1", "rows":[{"id":"q01", "answer":"previous"}]}))
    asyncio.run(module.generate("test-model", output, resume=True))
    generate.assert_awaited_once()
    rows = json.loads(output.read_text())["rows"]
    assert [r["id"] for r in rows] == ["q01", "q02"]
    assert rows[0]["answer"] == "previous"


def test_existing_answer_run_is_not_overwritten(tmp_path, monkeypatch):
    generate = setup_run(monkeypatch)
    output = tmp_path / "answers.json"
    output.write_text("existing")
    with pytest.raises(ValueError, match="Output exists"):
        asyncio.run(module.generate("test-model", output))
    assert output.read_text() == "existing"
    generate.assert_not_called()


def test_resume_rejects_different_model(tmp_path, monkeypatch):
    setup_run(monkeypatch)
    output = tmp_path / "answers.json"
    output.write_text(json.dumps({"model":"another-model", "dataset_sha256":module.fingerprint(), "rows":[]}))
    with pytest.raises(ValueError, match="different model or dataset"):
        asyncio.run(module.generate("test-model", output, resume=True))


@pytest.mark.parametrize("updates,match", [
    ({"answer_protocol":"freeform"}, "different answer protocol"),
    ({"rows":[{"id":"q01"}, {"id":"q01"}]}, "Invalid question IDs"),
    ({"rows":[{"id":"q99"}]}, "Invalid question IDs"),
])
def test_resume_rejects_incompatible_checkpoint(tmp_path, monkeypatch, updates, match):
    generate = setup_run(monkeypatch)
    output = tmp_path / "answers.json"
    report = {"model":"test-model", "dataset_sha256":module.fingerprint(),
              "answer_protocol":"source_ids_v1", "rows":[]}
    report.update(updates)
    output.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        asyncio.run(module.generate("test-model", output, resume=True))
    generate.assert_not_called()
