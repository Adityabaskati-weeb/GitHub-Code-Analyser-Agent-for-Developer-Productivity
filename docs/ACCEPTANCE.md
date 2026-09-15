# Local repository QA acceptance decision

Decision date: 2026-09-15. Status: the frozen `code_compact` authored run was measured and fails the fully-supported gate; cross-repository and matched-latency acceptance remain unassessed.

The project goal is a defensible, resume-worthy local repository question-answering system with measured answer quality, source provenance, operational reliability, and reproducible evaluation. This is not an expert security reviewer, whole-program verifier, or demonstrated developer-productivity improvement. The technical lead sets the design and acceptance criteria; the implementation worker implements and verifies changes.

The frozen compact authored run (`evaluation/code-compact-sourceid-full.json`) scored 28/50 fully supported, 14 partial, 8 zero, mean 1.40/2, 7 incorrect delivered answers, 1 false abstention, and 49/50 answer coverage. The mean, incorrect-delivery, and false-abstention thresholds passed; the required 30/50 fully-supported threshold failed. Astra’s row-level model-assisted review is `evaluation/code-compact-sourceid-full-astra-review.json`. No cross-repository or matched-load latency acceptance claim follows from this result.

## Constraints and existing evidence

Use the installed `qwen2.5-coder:3b` model. No new model downloads, paid model API calls by the application, or fabricated results. Application generation must remain local: do not add external model services or send reference answers to the generator. The generator receives questions and retrieved source only; reference answers and benchmark annotations are evaluator inputs. This restriction does not prohibit the user-authorized Codex agents from reading source and reference answers for explicitly declared model-assisted evaluation; that review is not independent human grading and must be labeled accordingly.

The published authored-set baseline has 19/50 fully supported answers, mean 1.16/2, 11 incorrect answers and no abstentions. Its lexical retrieval found the labeled file for 44/50 questions and source anchor for 39/50. The new code retrieval configuration finds 50/50 files and 47/50 anchors, but those are retrieval coverage results, not answer accuracy.

The interrupted `quoted_v2` candidate must finish under its existing configuration and be graded before further model experiments. Its observed responses show both false abstentions from inaccurate copied quotes and incorrect prose accompanied by valid quotes. Exact-quote validation establishes provenance, not semantic support. Preserve this candidate, its raw outputs, and its failures.

## Fixed quality targets

These are aspirational acceptance gates. Do not lower them after observing a candidate. If they remain unmet, document the unmet gates and describe the system as an evaluated prototype.

| Evaluation | Required result |
|---|---|
| Authored 50-question regression set | At least 30/50 fully supported answers; mean at least 1.40/2 |
| Incorrect delivered answers on authored set | At most 8 |
| False abstentions on authored answerable set | At most 5 |
| Existing retrieval configurations | Preserve lexical regression floors and code configuration's 50/50 file and 47/50 anchor coverage |
| Fresh cross-repository evaluation | Two distinct pinned repositories; each contributes 12 answerable and 4 absent-evidence questions |
| Cross-repository answerable questions | At least 15/24 fully supported; mean at least 1.40/2; at most 4 incorrect delivered answers |
| Cross-repository abstention | At least 7/8 correct absent-evidence abstentions; at most 3 false abstentions on answerable questions |

Grade with the existing 0/1/2 source-based rubric and explicit reviewer declaration. Wrong return types, contradictory behavior, unsupported assertions, generation failures, and false abstentions cannot receive full credit. Overall denominators include failed responses. Report answer coverage beside any answered-only accuracy. Report the two cross-repository results separately as well as pooled; this small sample does not establish general accuracy.

Freeze fresh questions, source revisions, reference answers, and dataset hashes before the final candidate runs. Use existing accessible source where possible, verify licensing before publishing excerpts, and include distractor files, conditional behavior, and some questions requiring multiple excerpts. Sources must differ from the development repository. Any subsequent tuning on those results converts that set into development data; do not retain a held-out claim.

## Implementation sequence

The fixed development pilot IDs are `q03, q07, q11, q15, q19, q23, q28, q32, q36, q40, q44, q48`, in that order. Selection uses the midpoint of each of twelve equal-width intervals across the existing 50 questions: one-based position `floor((i + 0.5) * 50 / 12) + 1` for `i = 0..11`. This systematic rule distributes questions across the file-grouped dataset without selecting on observed successes or failures. Existing outputs were already inspected before this decision, so the pilot is explicitly development data, not blind or held out. Both candidate pilots use exactly these IDs. The machine-readable selection is `evaluation/pilot_selection.json`.

The two selected fresh repositories are [tkem/cachetools](https://github.com/tkem/cachetools), for cache eviction, expiry and decorator behavior, and [theskumar/python-dotenv](https://github.com/theskumar/python-dotenv), for parsing, interpolation and environment-loading behavior. They provide compact Python source with different domains. The upstream [cachetools license](https://raw.githubusercontent.com/tkem/cachetools/master/LICENSE) is MIT; the [python-dotenv license](https://raw.githubusercontent.com/theskumar/python-dotenv/main/LICENSE) is BSD-style with three conditions. Preserve each complete license and copyright notice with published source fixtures. After the candidate is frozen, resolve each repository to an exact commit SHA, fetch that snapshot, verify the snapshot's license, and record source checksums before authoring the 32 questions. Do not use a moving branch name as the evaluation identity. Do not substitute a repository after observing its answer results.

1. Recover and complete the existing quoted candidate unchanged, then review every answer.
2. Run a fixed development pilot with code retrieval and existing `source_ids_v1`, retaining K=8 and the 18,000-character budget to isolate the quotation protocol.
3. If needed, implement one generic compact-context candidate: retrieve eight candidates, select three or four useful syntax excerpts within roughly 6,000–8,000 characters, and render citations from validated source IDs. Preserve short functions, signatures, branches, and necessary enclosing context. Do not hardcode questions, references, or repository-specific answers.
4. Measure retrieval coverage after the candidate's actual context selection. Do not attribute K=8 coverage to a smaller prompt without measuring it.
5. Select and freeze the candidate using development evidence, then run complete authored and cross-repository evaluations sequentially. Preserve unsuccessful pilots and identify the selected configuration explicitly.

Quotation validation may remain an experimental diagnostic. Do not weaken it and market the resulting acceptance rate as truth. A short answer instruction should request the mechanism, value, type, and units when relevant without forcing all questions into one sentence.

## Bounded inference and latency

For the next experiment cycle, allow at most 160 new logical generation requests: 11 remaining old-candidate rows, at most two fixed 12-question development pilots, 50 final authored questions, 32 cross-repository questions, up to 24 matched latency comparison requests, and at most 19 explicit failed-request reruns. Reuse final or pilot requests for timing where possible. No concurrent local inference jobs. Record every actual HTTP inference attempt; with at most one transport retry per logical request, the maximum is 320 attempts. A failed request still consumes budget. Exhaustion requires a documented technical reassessment, not hidden additional runs or silently relaxed gates.

Retain temperature 0, 8192 context tokens and maximum 1024 generated tokens unless a separately versioned experiment explicitly changes them. Record model digest, runtime version, hardware, timeout, retry policy, input/output token counts, generation calls, and elapsed time. Zero paid API spend does not mean zero compute cost; do not invent electricity or monetary estimates.

Report median and nearest-rank P95 generation latency, retrieval latency, and total call time. Historical baseline timings included concurrent work and cannot gate a new isolated run. A latency comparison requires the same hardware, runtime, model digest, question subset, warm/cold policy, and background-load conditions; alternate baseline and candidate order where practical and record the sample size. Under such matched conditions, target no more than a 10% increase in warm median and P95 versus the baseline. Without a matched comparison, latency acceptance is unassessed; publish descriptive timings only. There is no absolute cross-hardware latency gate.

## Reliability and source boundaries

- Persist each requested evaluation outcome atomically, including explicit transport failure, malformed response, truncation, no evidence, and valid answer states. Never silently omit failed rows or overwrite a completed run.
- Resume only an identical dataset, model, prompt/protocol, retrieval/context configuration, and relevant implementation. Store a clean revision or implementation digest plus any patch digest needed to distinguish dirty-worktree runs.
- Propagate per-response Ollama token usage, timing, termination information, and inference-attempt counts to answer metrics. Avoid mutable global counters.
- Keep strict source-ID/citation checks. Provenance tests do not substitute for semantic answer review.
- Enforce resolved local-source containment before traversal and file reads. On Windows, rejecting symbolic links alone does not necessarily exclude junctions; test that links or junctions cannot lead outside the selected root. Parse inspected source without importing or executing it.
- Verify configuration-change resume rejection, checkpoint recovery, malformed/truncated output handling, source-boundary enforcement, and absent-evidence behavior. Run relevant existing tests and required CI checks. Model-backed CI is not required.

## Reporting and handoff

Publish dataset fingerprints, source revisions, configuration identity, raw model outputs, retrieved excerpts, reviewer reasons, complete result counts, token/call costs, and reproducible commands. Keep baseline and candidate configurations distinct in the README and include the code retrieval option. Retain failed experiments and document remaining errors. A resume claim may describe the implemented local QA system and its measured evaluation improvement only when the completed artifacts support that wording.
