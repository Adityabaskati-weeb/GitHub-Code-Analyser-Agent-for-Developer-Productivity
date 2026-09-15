# Evidence-quality experiment

## What changed

The original `lexical` mode and its 50-question artifacts remain reproducible. The new `code` mode combines lightweight identifier tokenization with BM25 scoring and bounded Python syntax chunks. CamelCase and snake_case identifiers are split into words; simple suffix normalization and acronym candidates help connect questions to code. This is a heuristic tokenizer, not a full language model or linguistic stemmer.

Python functions, async functions and methods are retained as complete units up to 100 lines. Longer functions use overlapping windows. Module/class-level statements and comments remain in gap windows. Invalid Python, other languages and parsed-only inputs use the original line-window fallback. Source is parsed, never imported or executed. Class attributes and callers can still be separated from methods; this is not whole-program analysis.

The experimental `quoted_v2` path uses local Ollama and requires short exact source quotes. Quotes are checked against retrieved text, ignoring whitespace differences only. A mistaken source ID can be corrected only when the exact quote uniquely matches another retrieved excerpt; corrections are recorded. Unknown IDs, ambiguous relocation and invented quotes fail validation. Invalid evidence causes abstention, while malformed/truncated output is retained and reported as a failed response. The selected compact candidate uses the simpler versioned `source_ids_v1` protocol; the application renders locations from validated IDs.

This checks quote provenance, not semantic entailment. An exact quote can still be irrelevant to the answer. Requiring every quote to validate can also reject an otherwise useful answer; answer grading counts those abstentions as failures on the answerable regression set.

## Retrieval comparison

Same frozen 33-file corpus, 50 questions, K=8 and 18,000-character budget:

| Retrieval mode | Expected-file hits | Anchor hits | MRR@8 |
|---|---:|---:|---:|
| Original lexical / 40-line windows | 44/50 (88%) | 39/50 (78%) | 0.603667 |
| Identifier BM25 / Python syntax units | 50/50 (100%) | 47/50 (94%) | 0.857333 |

This is an authored development set, not held out. The new implementation was evaluated as one retrieval candidate; no question-specific answer mappings were added. The remaining anchor misses are q01, q22 and q31. All three still retrieve the labeled file. Full per-question results are in `evaluation/code_results.json`. Separate baseline files gate both modes in CI; the original thresholds are unchanged.

## Compact context candidate

The selected development candidate is `code_compact`: it keeps the raw `code` retriever at K=8, then selects at most four complete ranked excerpts within an 8,000-character rendered-source budget. `compact_v1` removes duplicate or contained same-file spans without round-robin re-ranking or character-prefix truncation, so every retained source ID keeps truthful original line ranges. Its versioned `source_ids_v1` prompt requests the exact mechanism, values, return type, units and conditions established by the selected excerpts in one to three concise sentences. The 8,000-character limit covers rendered source blocks and separators; prompt/schema and `Source ID` labels are reported separately. This is a context-and-prompt change, not a claim that the compact context preserves the raw K=8 coverage.

The offline coverage report measures the actual compact selection, not raw K=8 retrieval: `evaluation/code_compact_results.json` records 0.96 file recall, 0.82 evidence-anchor coverage, and a maximum of four selected excerpts/7,672 rendered characters. Reproduce it with:

```bash
python -m evaluation.run --mode code_compact --output evaluation/code_compact_results.json
python run_cli.py src --local --retrieval-mode code_compact --retrieval-only -q "How is invalid JSON handled?"
```

```bash
python -m evaluation.run --check
python -m evaluation.run --mode code --check
python run_cli.py src --local --retrieval-mode code --retrieval-only -q "How is invalid JSON handled?"
python run_cli.py src --local --retrieval-mode code -q "How is invalid JSON handled?"
```

## Answer-quality validation

```bash
python -m evaluation.answers --retrieval-mode code --output reports/code-answer-review.json
python -m evaluation.answers --retrieval-mode code --resume --output reports/code-answer-review.json
```

The original answers use `source_ids_v1`; the new candidate jointly changes retrieval and prompting. Comparing them measures the combined pipeline change, not either component's isolated causal effect. The same 0/1/2 rubric and explicit model-assisted reviewer declaration apply. Partial runs cannot be scored as full-benchmark accuracy.

The fixed 12-question `code` + `source_ids_v1` development pilot scored 4/12 fully supported, 6 partial, 2 incorrect, mean 1.1667/2, with no generation failures or abstentions. Its raw run and review are preserved in `evaluation/code-sourceid-pilot.json` and `evaluation/code-sourceid-pilot_review.json`.

The compact pilot scored 5/12 fully supported, 6 partial, 1 incorrect, mean 1.3333/2, with no generation failures or abstentions. Its raw run and review are preserved in `evaluation/code-compact-sourceid-pilot.json` and `evaluation/code_compact_pilot_review.json`.

After freezing the compact candidate unchanged, the complete 50-question authored run scored 28/50 fully supported, 14 partial, 8 incorrect/zero, mean 1.40/2, 7 incorrect delivered answers, 1 false abstention, and 49/50 answer coverage. Median/P95 generation time was 16.658/29.266 seconds; retrieval time was 25.581/40.386 ms. These timings are descriptive only: the historical baseline used different load conditions, so no matched latency gate was run. The fixed authored gate requires at least 30 fully supported answers, so the candidate is not acceptance-ready. The raw run, complete scored artifact, and Astra’s row-level review are preserved in `evaluation/code-compact-sourceid-full.json`, `evaluation/code-compact-sourceid-full-reviewed.json`, and `evaluation/code-compact-sourceid-full-astra-review.json`. The improvement over the original published baseline (19/50 full, mean 1.16/2, 11 incorrect) is a model-assisted source review on one authored set, not independent human or held-out validation.

Early quoted-output pilots revealed truncation, verbose explanations and source-ID mistakes. The completed `quoted_v2` full run scored 16/50 fully supported, 11 partial, 23 zero, mean 0.86/2, and 17 invalid-quote abstentions; its raw answers and model-assisted review remain in `evaluation/code_answers.json`. The prompt was shortened and given the schema explicitly, malformed-output abstention was added, and unique exact-quote relocation was implemented before that run. The result is preserved as a failed experiment, not a blind model comparison.

## Additional diagnostics

```bash
python -m evaluation.oracle --output reports/oracle.json
python -m evaluation.challenges --output reports/challenges.json
```

The oracle diagnostic intentionally bypasses retrieval using the labeled file and anchor for the 11 originally incorrect answers. It still calls the real local model with the original answer protocol. The model receives the question and gold source excerpt, not the reference answer. Oracle scores must never be advertised as deployable retrieval accuracy.

The eight synthetic challenge questions include four answerable questions and four whose answers are absent from the supplied source. They test units, return types, exceptions and refusal behavior. The set was authored after the implementation but before model execution; it is a small smoke set, not independent cross-repository validation. Only its separate results can support claims about unanswerable questions.
