"""Generate local model answers for declared review, or score a completed rubric."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import warnings
from time import perf_counter

from langchain_core.messages import HumanMessage
from evaluation.run import load_dataset, dataset_fingerprint
from src.config.settings import get_llm
from src.nodes.summarize_repo_node import summarize_repo_node
from src.retrieval.context_selector import (
    COMPACT_CONTEXT_BUDGET,
    COMPACT_CONTEXT_VERSION,
    COMPACT_MAX_EXCERPTS,
)

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_BUDGET_CHARS = 18000
PROMPT_VERSION = "summarize_repo_node.prompt.v2"
PROMPT_VERSIONS = {"code_compact": "summarize_repo_node.prompt.compact_v1"}
IDENTITY_FILES = (
    "evaluation/answers.py",
    "evaluation/run.py",
    "src/config/settings.py",
    "src/llm/ollama.py",
    "src/nodes/summarize_repo_node.py",
    "src/retrieval/citations.py",
    "src/retrieval/chunker.py",
    "src/retrieval/code_search.py",
    "src/retrieval/context_selector.py",
    "src/retrieval/evidence.py",
    "src/retrieval/ranker.py",
    "src/retrieval/quoted_answer.py",
    "src/retrieval/structured_answer.py",
)
PILOT_SELECTION_PATH = ROOT / "evaluation" / "pilot_selection.json"


def fingerprint():
    return dataset_fingerprint()


def implementation_fingerprint():
    """Hash the implementation that determines retrieval, prompting, and rendering."""
    digest = hashlib.sha256()
    for relative in IDENTITY_FILES:
        path = ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def prompt_fingerprint(mode, protocol):
    digest = hashlib.sha256(
        f"{PROMPT_VERSIONS.get(mode, PROMPT_VERSION)}|retrieval={mode}|protocol={protocol}".encode("utf-8")
    )
    for relative in (
        "src/nodes/summarize_repo_node.py",
        "src/retrieval/quoted_answer.py",
        "src/retrieval/structured_answer.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def hardware_metadata():
    """Return descriptive host information without probing or changing runtime state."""
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def select_questions(questions, question_ids=None, pilot=False):
    """Select an explicit, ordered subset while rejecting unknown/duplicate IDs."""
    if pilot and question_ids:
        raise ValueError("Use --pilot or --question-ids, not both")
    if pilot:
        selection = json.loads(PILOT_SELECTION_PATH.read_text(encoding="utf-8"))
        question_ids = selection.get("question_ids")
        if not isinstance(question_ids, list) or not question_ids:
            raise ValueError("Pilot selection must declare ordered question_ids")
    if question_ids is None:
        return list(questions)
    ids = list(question_ids)
    known = {question["id"]: question for question in questions}
    if len(ids) != len(set(ids)) or any(question_id not in known for question_id in ids):
        raise ValueError("Unknown or duplicate question IDs")
    return [known[question_id] for question_id in ids]


async def _runtime_metadata(llm):
    getter = getattr(llm, "aget_run_metadata", None)
    if not callable(getter):
        return {}
    try:
        metadata = await getter()
    except Exception:  # noqa: BLE001 - diagnostics must not block local inference
        return {}
    return metadata if isinstance(metadata, dict) else {}


def build_run_config(llm, mode, protocol, runtime_metadata=None, question_ids=None):
    """Return the immutable identity used to accept or reject a resume."""
    runtime_metadata = runtime_metadata or {}
    timeout = getattr(llm, "timeout", os.getenv("OLLAMA_TIMEOUT", "120"))
    retries = getattr(llm, "retries", 1)
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        timeout = 120.0
    try:
        retries = int(retries)
    except (TypeError, ValueError):
        retries = 1
    model_digest = runtime_metadata.get("model_digest")
    runtime_version = runtime_metadata.get("runtime_version")
    context_config = {
        "context_selector": "bounded_v1",
        "context_budget_chars": CONTEXT_BUDGET_CHARS,
        "max_context_excerpts": 8,
    }
    if mode == "code_compact":
        context_config = {
            "context_selector": COMPACT_CONTEXT_VERSION,
            "context_budget_chars": COMPACT_CONTEXT_BUDGET,
            "max_context_excerpts": COMPACT_MAX_EXCERPTS,
        }
    return {
        "schema_version": 1,
        "model": getattr(llm, "model", "unknown"),
        "model_digest": model_digest or "unknown",
        "runtime_version": runtime_version or "unknown",
        "identity_verified": bool(model_digest and runtime_version),
        "retrieval_mode": mode,
        "answer_protocol": protocol,
        "question_ids": list(question_ids or []),
        "top_k": 8,
        **context_config,
        "prompt_sha256": prompt_fingerprint(mode, protocol),
        "implementation_sha256": implementation_fingerprint(),
        "ollama": {
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 1024,
            "timeout_seconds": timeout,
            "max_retries": retries,
        },
        "python": platform.python_version(),
    }


def _atomic_write(output, report):
    """Durably checkpoint a report, replacing the destination only when complete."""
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(output.suffix + ".tmp")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, output)


def _failure_status(exc):
    message = str(exc).lower()
    if "malformed" in message or "invalid structured" in message:
        return "malformed_response"
    if "truncated" in message or "incomplete" in message:
        return "truncated_response"
    if hasattr(exc, "metadata") or "ollama" in message or "timed out" in message:
        return "transport_failure"
    return "generation_failure"


def _answer_outcome(metrics):
    """Classify a delivered result without conflating provenance and truth."""
    if not metrics.get("generated"):
        return "no_evidence" if metrics.get("retrieved_chunks", 0) == 0 else "retrieval_only"
    if metrics.get("done") is False or metrics.get("termination_reason") in {
        "length", "max_tokens", "limit", "incomplete"
    }:
        return "truncated_response"
    if metrics.get("abstained"):
        quote_status = (metrics.get("quote_check") or {}).get("status")
        if quote_status == "invalid_quote":
            return "invalid_quote_abstention"
        if quote_status == "malformed_response":
            return "malformed_response"
        if quote_status == "abstained":
            return "insufficient_evidence_abstention"
        return "abstained"
    return "answered"


def _failure_row(question, exc, protocol, mode, total_elapsed_ms):
    metadata = getattr(exc, "metadata", {})
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    outcome = _failure_status(exc)
    captured_sources = getattr(exc, "_qa_sources", [])
    captured_output = getattr(exc, "_qa_model_output", "")
    captured_context_chars = getattr(exc, "_qa_context_chars", 0)
    captured_chunks = getattr(exc, "_qa_retrieved_chunks", len(captured_sources))
    captured_retrieval_ms = getattr(exc, "_qa_retrieval_ms", None)
    captured_generation_ms = getattr(exc, "_qa_generation_ms", None)
    captured_total_call_ms = getattr(exc, "_qa_total_call_ms", total_elapsed_ms)
    captured_selection = getattr(exc, "_qa_selection", {})
    captured_selection = dict(captured_selection or {})
    captured_inference = getattr(exc, "_qa_inference", metadata)
    captured_inference = dict(captured_inference or {})
    metadata = {**metadata, **captured_inference}
    attempts = metadata.get("inference_attempts", 0)
    return {
        "id": question["id"],
        "question": question["question"],
        "reference_answer": question["answer"],
        "answer": "No answer was accepted because local inference failed.",
        "sources": captured_sources,
        "metrics": {
            "retrieval_ms": captured_retrieval_ms,
            "generation_ms": captured_generation_ms,
            "total_call_ms": round(captured_total_call_ms, 3),
            "context_chars": captured_context_chars,
            "retrieved_chunks": captured_chunks,
            "citation_check": {"status": "not_applicable", "checked": 0, "invalid": []},
            "abstained": False,
            "answer_protocol": protocol,
            "quote_check": {"status": "not_applicable", "quotes_checked": 0},
            "retrieval_mode": mode,
            "generated": False,
            "inference": metadata,
            "inference_attempts": attempts,
            "termination_reason": metadata.get("termination_reason"),
            "outcome": outcome,
            **captured_selection,
        },
        "model_output": captured_output,
        "status": "failed",
        "outcome": outcome,
        "error": {"type": type(exc).__name__, "message": str(exc)},
        "score": None,
        "reviewer_reason": "",
    }


def _ensure_totals(report):
    rebuild = "totals" not in report
    totals = report.setdefault("totals", {})
    defaults = {
        "logical_requests": int(report.get("logical_requests", 0) or 0),
        "generation_calls": 0,
        "inference_attempts": int(report.get("inference_attempts", 0) or 0),
        "failed_requests": 0,
        "retrieval_ms": 0.0,
        "generation_ms": 0.0,
        "total_call_ms": 0.0,
    }
    for key, value in defaults.items():
        totals.setdefault(key, value)
    if rebuild and report.get("rows"):
        # Legacy reports predate aggregate counters; reconstruct honestly from
        # the rows instead of presenting only the resumed suffix as the total.
        totals.update({key: 0 for key in (
            "logical_requests", "generation_calls", "inference_attempts",
            "failed_requests")})
        totals.update({key: 0.0 for key in ("retrieval_ms", "generation_ms", "total_call_ms")})
        for row in report["rows"]:
            metrics = row.get("metrics", {})
            totals["logical_requests"] += 1
            totals["generation_calls"] += int(bool(metrics.get("inference_attempts", 0)))
            totals["inference_attempts"] += int(metrics.get("inference_attempts", 0) or 0)
            totals["failed_requests"] += int(row.get("status") == "failed")
            for key in ("retrieval_ms", "generation_ms", "total_call_ms"):
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    totals[key] += value
    report["logical_requests"] = totals["logical_requests"]
    report["inference_attempts"] = totals["inference_attempts"]
    return totals


def _record_totals(report, row):
    totals = _ensure_totals(report)
    metrics = row.get("metrics", {})
    totals["logical_requests"] += 1
    totals["generation_calls"] += int(bool(metrics.get("inference_attempts", 0)))
    totals["inference_attempts"] += int(metrics.get("inference_attempts", 0) or 0)
    totals["failed_requests"] += int(row.get("status") == "failed")
    for key in ("retrieval_ms", "generation_ms", "total_call_ms"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            totals[key] += value
    report["logical_requests"] = totals["logical_requests"]
    report["inference_attempts"] = totals["inference_attempts"]


async def generate(model, output, resume=False, mode="lexical", protocol=None,
                   allow_legacy_resume=False, allow_unverified_resume=False,
                   question_ids=None, pilot=False):
    corpus, all_questions = load_dataset()
    questions = select_questions(all_questions, question_ids, pilot)
    llm = get_llm("ollama", model)
    protocol = protocol or ("quoted_v2" if mode == "code" else "source_ids_v1")
    runtime_metadata = await _runtime_metadata(llm)
    run_config = build_run_config(llm, mode, protocol, runtime_metadata,
                                  [q["id"] for q in questions])
    now = datetime.now(timezone.utc).isoformat()
    selected_ids = [q["id"] for q in questions]
    report = {"model":llm.model, "dataset_sha256":fingerprint(), "reviewer_type":"unassigned",
              "answer_protocol":protocol, "retrieval_mode":mode,
              "rubric":"0 incorrect/unsupported, 1 partially correct, 2 correct and supported; grade all 50.",
              "source_revision": corpus.get("revision"), "question_ids": selected_ids,
              "run_config": run_config, "hardware": hardware_metadata(),
              "runtime": runtime_metadata,
              "run_status": "in_progress", "started_at_utc": now,
              "completed_at_utc": None, "resumed_at_utc": None, "resume_count": 0,
              "logical_requests": 0, "inference_attempts": 0,
              "totals": {"logical_requests": 0, "generation_calls": 0,
                         "inference_attempts": 0, "failed_requests": 0,
                         "retrieval_ms": 0.0, "generation_ms": 0.0,
                         "total_call_ms": 0.0}, "rows":[]}
    if output.exists():
        if not resume:
            raise ValueError("Output exists; choose a new path or pass --resume")
        report = json.loads(output.read_text(encoding="utf-8"))
        if report["model"] != llm.model or report["dataset_sha256"] != fingerprint():
            raise ValueError("Cannot resume answers from a different model or dataset")
        ids = [r.get("id") for r in report.get("rows", [])]
        if (len(ids) != len(set(ids)) or any(qid not in {q["id"] for q in questions} for qid in ids)):
            raise ValueError("Invalid question IDs in partial report")
        if report.get("answer_protocol") != protocol:
            raise ValueError("Cannot resume a different answer protocol")
        if report.get("retrieval_mode", "lexical") != mode:
            raise ValueError("Cannot resume a different retrieval mode")
        if "run_config" not in report:
            if not allow_legacy_resume:
                raise ValueError(
                    "Cannot resume a legacy report without --allow-legacy-resume"
                )
            warnings.warn(
                "Resuming an unfingerprinted legacy report; reproducibility is limited.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif report["run_config"] != run_config:
            raise ValueError("Cannot resume with a different run configuration")
        elif not report["run_config"].get("identity_verified", False):
            if not allow_unverified_resume:
                raise ValueError(
                    "Cannot resume without verified runtime/model identity; "
                    "use --allow-unverified-resume only for an explicitly unverified run"
                )
            warnings.warn(
                "Resuming with unverified runtime/model identity; do not promote this run.",
                RuntimeWarning,
                stacklevel=2,
            )
        stored_ids = report.get("question_ids", report.get("run_config", {}).get("question_ids"))
        if stored_ids is not None and list(stored_ids) != selected_ids:
            raise ValueError("Cannot resume with a different question selection")
        report["resumed_at_utc"] = now
        report["resume_count"] = int(report.get("resume_count", 0)) + 1
        _ensure_totals(report)
    completed = {r["id"] for r in report["rows"]}
    if not output.exists():
        _atomic_write(output, report)
    _ensure_totals(report)
    for question in questions:
        if question["id"] in completed:
            continue
        request_started = perf_counter()
        try:
            result = await summarize_repo_node({"parsed_files":corpus["files"],
                "messages":[HumanMessage(content=question["question"])], "llm":llm,
                "top_k":8, "retrieval_mode":mode, "answer_protocol":protocol})
            metrics = dict(result.get("metrics", {}))
            outcome = _answer_outcome(metrics)
            metrics["outcome"] = outcome
            row = {"id":question["id"], "question":question["question"],
                "reference_answer":question["answer"], "answer":result["summary"],
                "sources":result.get("sources", []), "metrics":metrics,
                "model_output":result.get("model_output", ""),
                "status":"completed",
                "outcome":outcome,
                "score":None, "reviewer_reason":""}
        except Exception as exc:  # noqa: BLE001 - persist every requested outcome
            row = _failure_row(question, exc, protocol, mode,
                               (perf_counter() - request_started) * 1000)
        report["rows"].append(row)
        _record_totals(report, row)
        _atomic_write(output, report)
        completed.add(question["id"])
        print(f"Saved answer {len(report['rows'])}/{len(questions)}", flush=True)
    report["run_status"] = "completed"
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(output, report)


def score(report):
    _, all_questions = load_dataset()
    configured_ids = ((report.get("run_config") or {}).get("question_ids")
                      if isinstance(report.get("run_config"), dict) else None)
    questions = select_questions(all_questions, configured_ids or None)
    rows = report["rows"]
    if report["dataset_sha256"] != fingerprint():
        raise ValueError("Answer review uses a different dataset")
    if report.get("reviewer_type") not in {"human", "model-assisted"}:
        raise ValueError("Declare reviewer_type as human or model-assisted")
    if len(rows) != len(questions) or {r["id"] for r in rows} != {q["id"] for q in questions}:
        raise ValueError(
            f"Grade {len(questions)} unique question IDs; partial reviews cannot be scored"
        )
    if any(type(r["score"]) is not int or r["score"] not in {0, 1, 2}
           or not r.get("reviewer_reason", "").strip() for r in rows):
        raise ValueError("Every answer needs a 0/1/2 score and a written reviewer reason")
    return {"questions":len(rows), "reviewer_type":report["reviewer_type"],
            "mean_score_out_of_2":statistics.mean(r["score"] for r in rows),
            "fully_correct_fraction":sum(r["score"] == 2 for r in rows) / len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--retrieval-mode", choices=["lexical", "code", "code_compact", "semantic"], default="lexical")
    parser.add_argument("--protocol", choices=["source_ids_v1", "quoted_v2"])
    parser.add_argument("--output", type=Path, default=Path("reports/answer-review.json"))
    parser.add_argument("--score", type=Path, help="Score a manually graded JSON report without a model")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted answer run")
    parser.add_argument("--pilot", action="store_true", help="Run the frozen 12-question development pilot")
    parser.add_argument("--question-ids", nargs="+", metavar="ID",
                        help="Ordered question IDs (space- or comma-separated)")
    parser.add_argument("--allow-legacy-resume", action="store_true",
                        help="Explicitly resume a pre-harness report without run identity metadata")
    parser.add_argument("--allow-unverified-resume", action="store_true",
                        help="Explicitly resume when local runtime/model identity could not be verified")
    args = parser.parse_args()
    if args.score:
        print(json.dumps(score(json.loads(args.score.read_text(encoding="utf-8"))), indent=2))
    else:
        question_ids = None
        if args.question_ids:
            if args.pilot:
                parser.error("Use --pilot or --question-ids, not both")
            question_ids = [item.strip() for value in args.question_ids
                            for item in value.split(",") if item.strip()]
        asyncio.run(generate(args.model, args.output, args.resume, args.retrieval_mode,
                             args.protocol, args.allow_legacy_resume,
                             args.allow_unverified_resume, question_ids, args.pilot))


if __name__ == "__main__":
    main()
