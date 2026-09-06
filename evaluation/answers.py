"""Generate local model answers for human grading, or score a completed rubric."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import statistics

from langchain_core.messages import HumanMessage
from evaluation.run import HERE, load_dataset, dataset_fingerprint
from src.config.settings import get_llm
from src.nodes.summarize_repo_node import summarize_repo_node


def fingerprint():
    return dataset_fingerprint()


async def generate(model, output):
    corpus, questions = load_dataset()
    llm = get_llm("ollama", model)
    report = {"model":llm.model, "dataset_sha256":fingerprint(),
              "rubric":"0 incorrect/unsupported, 1 partially correct, 2 correct and supported; grade all 50.",
              "rows":[]}
    output.parent.mkdir(parents=True, exist_ok=True)
    for question in questions:
        result = await summarize_repo_node({"parsed_files":corpus["files"],
            "messages":[HumanMessage(content=question["question"])], "llm":llm,
            "top_k":8, "retrieval_mode":"lexical"})
        report["rows"].append({"id":question["id"], "question":question["question"],
            "reference_answer":question["answer"], "answer":result["summary"],
            "sources":result["sources"], "metrics":result["metrics"],
            "score":None, "reviewer_reason":""})
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved answer {len(report['rows'])}/{len(questions)}")


def score(report):
    _, questions = load_dataset()
    rows = report["rows"]
    if report["dataset_sha256"] != fingerprint():
        raise ValueError("Answer review uses a different dataset")
    if len(rows) != len(questions) or {r["id"] for r in rows} != {q["id"] for q in questions}:
        raise ValueError("Grade all 50 unique question IDs; partial reviews cannot be scored")
    if any(type(r["score"]) is not int or r["score"] not in {0, 1, 2}
           or not r.get("reviewer_reason", "").strip() for r in rows):
        raise ValueError("Every answer needs a 0/1/2 score and a written reviewer reason")
    return {"questions":len(rows), "human_mean_score_out_of_2":statistics.mean(r["score"] for r in rows),
            "fully_correct_fraction":sum(r["score"] == 2 for r in rows) / len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/answer-review.json"))
    parser.add_argument("--score", type=Path, help="Score a manually graded JSON report without a model")
    args = parser.parse_args()
    if args.score:
        print(json.dumps(score(json.loads(args.score.read_text(encoding="utf-8"))), indent=2))
    else:
        asyncio.run(generate(args.model, args.output))


if __name__ == "__main__":
    main()
