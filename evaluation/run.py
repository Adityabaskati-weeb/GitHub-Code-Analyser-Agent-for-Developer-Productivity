"""Run with python -m evaluation.run. No model, credentials, or network needed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
from time import perf_counter

from src.retrieval.evidence import retrieve, bounded_context
from src.retrieval.context_selector import (
    COMPACT_CONTEXT_BUDGET,
    COMPACT_CONTEXT_VERSION,
    COMPACT_MAX_EXCERPTS,
    context_selection_metadata,
    rendered_context_chars,
    select_compact_context,
)

HERE = Path(__file__).resolve().parent


def dataset_fingerprint():
    # Normalize physical line endings so Windows and Linux grade the same bytes.
    return hashlib.sha256(b"".join((HERE / name).read_text(encoding="utf-8").encode("utf-8")
                                   for name in ("corpus.json", "questions.json"))).hexdigest()


def load_dataset():
    corpus = json.loads((HERE / "corpus.json").read_text(encoding="utf-8"))
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    files = corpus["files"]
    paths = {f["path"]: f for f in files}
    if len(paths) != len(files) or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate source paths or question IDs")
    if len(questions) != 50:
        raise ValueError("Expected the versioned 50-question benchmark")
    for f in files:
        if hashlib.sha256(f["raw"].encode()).hexdigest() != f["sha256"]:
            raise ValueError(f"Corpus checksum mismatch: {f['path']}")
    for q in questions:
        if q["path"] not in paths or q["anchor"] not in paths[q["path"]]["raw"]:
            raise ValueError(f"Missing source evidence for {q['id']}")
    return corpus, questions


def score_question(question, chunks):
    expected = question["path"]
    ranks = [i + 1 for i, c in enumerate(chunks) if c.path == expected]
    return {
        # Each question has one labeled relevant file, so file recall is binary.
        "file_recall": float(bool(ranks)),
        "reciprocal_rank": 1 / ranks[0] if ranks else 0,
        "evidence_hit": float(any(c.path == expected and question["anchor"] in c.text
                                  for c in chunks)),
    }


def evaluate(files, questions, top_k=8, chunk_lines=40, overlap=8, mode="lexical"):
    rows = []
    for q in questions:
        start = perf_counter()
        retrieval_mode = "code" if mode == "code_compact" else mode
        candidates = retrieve(q["question"], files, top_k, retrieval_mode, chunk_lines, overlap)
        if mode == "code_compact":
            chunks = select_compact_context(candidates, COMPACT_MAX_EXCERPTS,
                                            COMPACT_CONTEXT_BUDGET)
            selection = context_selection_metadata(
                chunks, candidates, COMPACT_MAX_EXCERPTS, COMPACT_CONTEXT_BUDGET)
        else:
            chunks = bounded_context(candidates)
            selection = {
                "context_selector": "bounded_v1",
                "context_budget_chars": 18000,
                "max_context_excerpts": top_k,
                "candidate_chunks": len(candidates),
                "selected_chunks": len(chunks),
                "selected_context_chars": rendered_context_chars(chunks),
            }
        elapsed = (perf_counter() - start) * 1000
        rows.append({"id": q["id"], "question": q["question"],
                     **score_question(q, chunks), **selection,
                     "retrieval_ms": round(elapsed, 3),
                     "sources": [f"{c.path}:L{c.start_line}-L{c.end_line}" for c in chunks]})
    times = sorted(r["retrieval_ms"] for r in rows)
    config = {"mode": mode, "top_k": top_k, "chunk_lines": chunk_lines,
              "overlap": overlap, "context_budget_chars": 18000}
    if mode == "code":
        config = {"mode":mode, "top_k":top_k, "chunking":"python_ast_v1",
                  "max_function_lines":100, "fallback_chunk_lines":40, "overlap":8,
                  "ranking":"identifier_bm25_v1", "context_budget_chars":18000}
    elif mode == "code_compact":
        config = {"mode":mode, "top_k":top_k, "retrieval":"code",
                  "chunking":"python_ast_v1", "max_function_lines":100,
                  "fallback_chunk_lines":40, "overlap":8,
                  "ranking":"identifier_bm25_v1",
                  "context_selector":COMPACT_CONTEXT_VERSION,
                  "max_excerpts":COMPACT_MAX_EXCERPTS,
                  "context_budget_chars":COMPACT_CONTEXT_BUDGET}
    return {
        "config": config,
        "summary": {"questions": len(rows),
                    **{key: round(statistics.mean(r[key] for r in rows), 6)
                       for key in ("file_recall", "reciprocal_rank", "evidence_hit")},
                    "median_retrieval_ms": round(statistics.median(times), 3),
                    "p95_retrieval_ms": times[max(0, __import__('math').ceil(len(times) * .95) - 1)]},
        "rows": rows,
    }


def check_baseline(result, baseline, dataset_hash):
    if baseline["dataset_sha256"] != dataset_hash:
        raise ValueError("Dataset changed: review and regenerate the baseline explicitly")
    if result["config"] != baseline["config"]:
        raise ValueError("Benchmark configuration differs from the approved baseline")
    failures = []
    for key, floor in baseline["minimums"].items():
        if result["summary"][key] + 1e-9 < floor:
            failures.append(f"{key}: {result['summary'][key]:.4f} < {floor:.4f}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare 20- and 40-line chunks at k=5 and k=8")
    parser.add_argument("--mode", choices=["lexical", "semantic", "code", "code_compact"], default="lexical")
    args = parser.parse_args()
    if args.compare and args.mode not in {"lexical", "semantic"}:
        parser.error("--compare varies line windows; use lexical or semantic, not AST mode")
    if args.check and args.mode == "code_compact":
        parser.error("code_compact is a measured candidate without an approved regression baseline; use --mode code --check for the preserved K=8 gate")
    corpus, questions = load_dataset()
    dataset_hash = dataset_fingerprint()
    result = evaluate(corpus["files"], questions, mode=args.mode)
    report = {"dataset_sha256": dataset_hash, "source_revision": corpus["revision"],
              "python": platform.python_version(), "platform": platform.system(),
              "scope": "Retrieval on a frozen single-repository corpus; not answer correctness or GitHub file selection.",
              **result}
    if args.compare:
        report["comparisons"] = [evaluate(corpus["files"], questions, k, lines, overlap, args.mode)
                                  for k, lines, overlap in ((5, 20, 4), (8, 20, 4), (5, 40, 8))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved {args.output}")
    if args.check:
        name = "baseline-code.json" if args.mode == "code" else "baseline.json"
        baseline = json.loads((HERE / name).read_text(encoding="utf-8"))
        failures = check_baseline(result, baseline, dataset_hash)
        if failures:
            print("REGRESSION: " + "; ".join(failures))
            return 1
        print("Retrieval regression gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
