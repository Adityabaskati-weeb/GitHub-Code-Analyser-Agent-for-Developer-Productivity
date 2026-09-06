# Evaluation methodology

## Reproduction

```bash
python -m evaluation.run --compare --output reports/comparison.json
python -m evaluation.run --check
```

The shipped corpus removes network availability, branch drift, hosted models and API quotas from retrieval CI. `evaluation/freeze_corpus.py` can reconstruct it from its pinned commit if that Git history is available. Normal benchmark runs only read the checked-in JSON files.

There are 50 authored developer questions and 33 source/config files from one earlier commit of this repository. Each question has one labeled relevant file, an exact source anchor, and a reference answer. Dataset validation verifies every file checksum and confirms every anchor exists. The questions and reference answers never enter retrieval context.

This is a regression dataset authored with knowledge of the source. It is not a held-out benchmark, does not test unanswerable queries, and does not measure the GitHub file-selection stage. It should not be advertised as independent evidence of hiring outcomes or general code-understanding accuracy.

## Metrics

- **File Recall@K:** fraction of questions where the labeled file appears in the selected chunks. Each question has one labeled file, so this is also file hit rate.
- **Evidence hit rate:** fraction where a retrieved chunk from the labeled file contains the exact answer anchor. This is a coverage proxy; matching an anchor does not prove the entire reference answer is supported.
- **MRR@K:** mean reciprocal rank of the first chunk from the labeled file, zero when absent.
- **Retrieval latency:** time to chunk, rank and fit the character budget for one question; includes rebuilding the lexical index. Excludes disk loading, GitHub fetching and model inference. Median and nearest-rank P95 are recorded. No hardware-independent speed claim is made.

Metrics are measured after the same context budget filter used by the app. The default is lexical TF-IDF cosine ranking, 40-line chunks, 8-line overlap, K=8, and an 18,000-character context budget. The checked-in artifact records the Python version, platform, dataset hash, source revision and per-question locations.

## Failure analysis

In the first recorded run, these questions lacked the annotated evidence: q07, q10, q16, q21, q23, q24, q31, q38, q39, q41 and q42. Five of them nevertheless found the expected file. For example, q10 retrieved the GitHub parser but not the exact default-branch method anchor.

Some misses are natural-language/code vocabulary mismatches: a question about inverse document frequency does not necessarily rank a short `_idf` implementation highly enough. Other misses reflect chunk boundaries or competing snippets. Candidate next experiments are identifier-aware tokenization and function-boundary chunks. Keep the current corpus fixed, report all configurations, and validate changes on additional repositories before drawing broader conclusions.

## Regression policy

CI requires file recall >= 0.88, evidence hit rate >= 0.78 and MRR >= 0.603667 for the default configuration. All are the observed initial values, not invented targets. Tests verify that lower scores fail and changed datasets are rejected. CI uploads the per-question result even if the evaluation gate fails.

Do not lower the baseline to hide a regression. If the dataset changes, document the reason, inspect the full results, update its hash and justify any new thresholds in the PR. Latency is not gated on shared CI hardware.

## Answer grading

```bash
python -m evaluation.answers --model qwen2.5-coder:3b
```

This runs local inference on all 50 questions and atomically checkpoints `reports/answer-review.json` after each answer. A missing or failed model raises an error; a partial report is retained for inspection but cannot be scored. Use `--resume` to continue with the same model and dataset. Existing output is never silently overwritten. The generator uses the same answer node and retrieval defaults as the application.

Declare `reviewer_type` as `human` or `model-assisted`. For each row, read the reference answer and frozen source, then enter a score and reviewer reason:

| Score | Meaning |
|---|---|
| 0 | Incorrect, unsupported, or fails to answer despite available evidence |
| 1 | Partially correct, but missing material information or usable citation support |
| 2 | Correct, sufficiently complete, and supported by accurate source citations |

```bash
python -m evaluation.answers --score reports/answer-review.json
```

The scorer checks the dataset hash, all unique question IDs, integer scores and written reasons. It reports the declared reviewer type, mean score out of two and fraction scored two. It cannot verify a reviewer's judgment. Model-assisted review is not human validation. Prefer an independent human reviewer and report agreement for stronger evidence. The included tests use synthetic grades only to validate arithmetic; those are not model results.

## Validation environment

The development machine's Windows sandbox has an ACL issue with Python temporary directories created with mode 0700 and a path-length limit. Local tests used a separate, uncommitted launcher with shorter workspace temp paths and inherited directory ACLs. It changes no assertions or application logic. Standard `python -m pytest tests -q` is the CI command on Linux.
