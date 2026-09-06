"""Generate local model answers for declared review, or score a completed rubric."""
import argparse
import asyncio
import json
from pathlib import Path
import statistics

from langchain_core.messages import HumanMessage
from evaluation.run import load_dataset, dataset_fingerprint
from src.config.settings import get_llm
from src.nodes.summarize_repo_node import summarize_repo_node


def fingerprint():
    return dataset_fingerprint()


async def generate(model, output, resume=False):
    corpus, questions = load_dataset()
    llm = get_llm("ollama", model)
    report = {"model":llm.model, "dataset_sha256":fingerprint(), "reviewer_type":"unassigned",
              "answer_protocol":"source_ids_v1",
              "rubric":"0 incorrect/unsupported, 1 partially correct, 2 correct and supported; grade all 50.",
              "rows":[]}
    if output.exists():
        if not resume:
            raise ValueError("Output exists; choose a new path or pass --resume")
        report = json.loads(output.read_text(encoding="utf-8"))
        if report["model"] != llm.model or report["dataset_sha256"] != fingerprint():
            raise ValueError("Cannot resume answers from a different model or dataset")
        if report.get("answer_protocol") != "source_ids_v1":
            raise ValueError("Cannot resume a different answer protocol")
        ids = [r["id"] for r in report["rows"]]
        if len(ids) != len(set(ids)) or not set(ids) <= {q["id"] for q in questions}:
            raise ValueError("Invalid question IDs in partial report")
    completed = {r["id"] for r in report["rows"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    for question in questions:
        if question["id"] in completed:
            continue
        result = await summarize_repo_node({"parsed_files":corpus["files"],
            "messages":[HumanMessage(content=question["question"])], "llm":llm,
            "top_k":8, "retrieval_mode":"lexical"})
        report["rows"].append({"id":question["id"], "question":question["question"],
            "reference_answer":question["answer"], "answer":result["summary"],
            "sources":result["sources"], "metrics":result["metrics"],
            "model_output":result.get("model_output", ""),
            "score":None, "reviewer_reason":""})
        pending = output.with_suffix(output.suffix + ".tmp")
        pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        pending.replace(output)
        print(f"Saved answer {len(report['rows'])}/{len(questions)}", flush=True)


def score(report):
    _, questions = load_dataset()
    rows = report["rows"]
    if report["dataset_sha256"] != fingerprint():
        raise ValueError("Answer review uses a different dataset")
    if report.get("reviewer_type") not in {"human", "model-assisted"}:
        raise ValueError("Declare reviewer_type as human or model-assisted")
    if len(rows) != len(questions) or {r["id"] for r in rows} != {q["id"] for q in questions}:
        raise ValueError("Grade all 50 unique question IDs; partial reviews cannot be scored")
    if any(type(r["score"]) is not int or r["score"] not in {0, 1, 2}
           or not r.get("reviewer_reason", "").strip() for r in rows):
        raise ValueError("Every answer needs a 0/1/2 score and a written reviewer reason")
    return {"questions":len(rows), "reviewer_type":report["reviewer_type"],
            "mean_score_out_of_2":statistics.mean(r["score"] for r in rows),
            "fully_correct_fraction":sum(r["score"] == 2 for r in rows) / len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/answer-review.json"))
    parser.add_argument("--score", type=Path, help="Score a manually graded JSON report without a model")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted answer run")
    args = parser.parse_args()
    if args.score:
        print(json.dumps(score(json.loads(args.score.read_text(encoding="utf-8"))), indent=2))
    else:
        asyncio.run(generate(args.model, args.output, args.resume))


if __name__ == "__main__":
    main()
