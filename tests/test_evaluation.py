import copy
import json

import pytest

from evaluation.run import HERE, load_dataset, score_question, check_baseline
from src.retrieval.evidence import Evidence


def test_every_golden_question_has_actual_source_evidence():
    corpus, questions = load_dataset()
    assert len(questions) == 50
    assert len(corpus["files"]) == 33


def test_file_hit_is_not_an_evidence_hit():
    q = {"path":"a.py", "anchor":"return 42"}
    result = score_question(q, [Evidence("a.py", 1, 1, "import os")])
    assert result["file_recall"] == 1
    assert result["evidence_hit"] == 0


def test_wrong_file_with_matching_text_does_not_count():
    result = score_question({"path":"a.py", "anchor":"return 42"},
                            [Evidence("b.py", 1, 1, "return 42")])
    assert result["file_recall"] == result["evidence_hit"] == 0


def test_regression_gate_actually_fails():
    baseline = json.loads((HERE / "baseline.json").read_text())
    result = {"config":baseline["config"], "summary":copy.deepcopy(baseline["minimums"])}
    assert check_baseline(result, baseline, baseline["dataset_sha256"]) == []
    result["summary"]["evidence_hit"] -= .02
    assert check_baseline(result, baseline, baseline["dataset_sha256"])


def test_changed_dataset_cannot_silently_pass_gate():
    baseline = json.loads((HERE / "baseline.json").read_text())
    with pytest.raises(ValueError, match="Dataset changed"):
        check_baseline({}, baseline, "different")


def test_unreviewed_answers_cannot_be_reported_as_accuracy():
    from evaluation.answers import score, fingerprint
    _, questions = load_dataset()
    report = {"dataset_sha256":fingerprint(), "rows":[
        {"id":q["id"], "score":None, "reviewer_reason":""} for q in questions]}
    with pytest.raises(ValueError, match="Every answer"):
        score(report)


def test_human_grade_calculation():
    from evaluation.answers import score, fingerprint
    _, questions = load_dataset()
    report = {"dataset_sha256":fingerprint(), "rows":[
        {"id":q["id"], "score":2 if i < 25 else 0, "reviewer_reason":"Test rubric fixture"}
        for i, q in enumerate(questions)]}
    result = score(report)
    assert result["human_mean_score_out_of_2"] == 1
    assert result["fully_correct_fraction"] == .5


def test_dataset_fingerprint_is_portable_across_line_endings(tmp_path, monkeypatch):
    import evaluation.run as module
    monkeypatch.setattr(module, "HERE", tmp_path)
    for name in ("corpus.json", "questions.json"):
        (tmp_path / name).write_bytes(b'{\n  "test": true\n}\n')
    before = module.dataset_fingerprint()
    for name in ("corpus.json", "questions.json"):
        (tmp_path / name).write_bytes(b'{\r\n  "test": true\r\n}\r\n')
    assert module.dataset_fingerprint() == before
