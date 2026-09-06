"""Audit the published run's provenance and arithmetic, not reviewer judgment."""
import json

from evaluation.run import HERE, load_dataset
from evaluation.answers import score
from src.retrieval.evidence import bounded_context, retrieve
from src.retrieval.structured_answer import render_answer


def test_published_review_matches_frozen_evidence_and_raw_outputs():
    report = json.loads((HERE / "live_answers.json").read_text(encoding="utf-8"))
    corpus, questions = load_dataset()
    by_id = {q["id"]:q for q in questions}
    assert report["reviewer_type"] == "model-assisted"
    for key, value in score(report).items():
        assert report["summary"][key] == value
    for row in report["rows"]:
        question = by_id[row["id"]]
        assert row["question"] == question["question"]
        assert row["reference_answer"] == question["answer"]
        chunks = bounded_context(retrieve(question["question"], corpus["files"], 8, "lexical"))
        assert row["sources"] == [c.to_dict() for c in chunks]
        catalog = {f"S{i}":c for i, c in enumerate(chunks, 1)}
        answer, abstained = render_answer(row["model_output"], catalog)
        assert row["answer"] == answer
        assert row["metrics"]["abstained"] == abstained
