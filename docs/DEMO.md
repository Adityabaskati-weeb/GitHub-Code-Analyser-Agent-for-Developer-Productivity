# A five-minute project walkthrough

## 1. Explain the user problem

"When I open an unfamiliar repository, I want answers tied to inspectable source code. This project retrieves source excerpts, lets a local model answer from them, and evaluates whether retrieval actually includes the relevant code."

## 2. Show the free evidence demo

```bash
python run_cli.py src --local --retrieval-only --no-report -q "How are cache filenames derived from URLs?"
```

Open the cited file and confirm the displayed line range. Explain that the output is retrieved evidence, not generated text. If Ollama is installed, repeat without `--retrieval-only` and inspect the generated answer and citations.

## 3. Reproduce the benchmark

```bash
python -m evaluation.run --check
```

Explain 44/50 expected-file hits versus 39/50 anchor hits. Show a failed question in `evaluation/results.json`. The failure cases are part of the result, not hidden from the report.

## 4. Defend one decision

Use the 20-line versus 40-line comparison. Smaller chunks achieved higher MRR but lower evidence coverage. Explain why the chosen objective was relevant source content rather than an impressive-looking rank score.

## 5. Show failure handling

```bash
python -m pytest tests/test_ollama.py tests/test_graph_integration.py tests/test_evaluation.py -q
```

Discuss missing local models, timeouts, retries, metadata failure, invalid inputs and a failed regression gate. Read these tests and run the examples yourself before presenting the project.

## Resume wording supported by the current evidence

"Extended a GitHub codebase RAG agent with local inference support, source-line provenance and automated retrieval regression checks; evaluated 50 annotated questions on a frozen repository corpus, achieving 88% file recall and 78% evidence-anchor coverage at K=8."

Use this only after you understand and can reproduce the work. Do not convert retrieval scores into answer accuracy, claim real-model validation that has not been run, or claim measured developer time savings without a user study.
