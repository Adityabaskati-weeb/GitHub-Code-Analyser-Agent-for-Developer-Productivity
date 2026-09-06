# Engineering decisions

## Local provider as the default

The original CLI required a Gemini API key. An explicit provider boundary now allows local Ollama inference and leaves Gemini optional. The local adapter uses the documented `/api/chat` endpoint with `stream=false`, a timeout, one bounded retry for transport/server failures, and actionable errors for missing models. The client ignores proxy environment settings for loopback requests. It does not download a model or fall back to a paid API.

Trade-off: local inference avoids per-request fees but speed and answer quality depend on hardware and model choice. A model-free evidence mode lets reviewers reproduce retrieval without that prerequisite.

## Preserve evidence instead of losing implementation details

The old summarization node built file labels but did not attach them to ranked chunks. Python parsing also retained names and docstrings while dropping bodies. Fetching now retains original source alongside parser output. Retrieval uses original source with file and line ranges; parsed-only fallback explicitly avoids claiming original source line numbers.

Trade-off: original code takes more context than summaries. Forty-line overlapping chunks and a character budget bound context size. Long-line truncation and function-boundary splitting remain limitations.

## Separate indexing from answering

The original indexing graph generated a global summary and a generic answer before the user asked a question. Indexing now fetches metadata only. Question answering selects and fetches files, retrieves evidence, then makes one model request (plus bounded retries if needed).

Trade-off: architecture questions no longer benefit from a speculative pre-generated overview. They are grounded in the actual retrieved source and may require more targeted questions.

## Explicit retrieval configuration

`top_k` and `retrieval_mode` are graph state fields. Previously, the CLI changed a settings attribute after modules had imported its old value, and `top_k` was absent from the graph schema. An integration test now traverses the compiled graph and checks the context cap.

Semantic mode raises a useful error when unavailable. Silent fallback made evaluation labels unreliable. The default lexical ranker computes document frequencies once per query instead of repeatedly scanning the corpus for every term. It also ranks small corpora and removes zero-overlap matches.

## Evidence hit rate alongside file recall

Both twenty- and forty-line chunks achieved 88% file recall at K=8 in this regression set. Forty-line chunks contained the annotated answer anchor on 78% of questions versus 60% with twenty-line chunks. Twenty-line chunks had higher MRR, so the trade-off is real: ranking a file earlier is not the same as supplying its useful code.

The default prioritizes evidence coverage. All four tested configurations and all misses remain in the result artifact. There is no fabricated before/after accuracy claim.

## Immutable evaluation inputs

The corpus is materialized from commit `24d5b624e881cc2ca66a3fd2664be1b2d2d8f5c8` and includes only source/config files, not this README, the answer key, or benchmark implementation. File hashes and a combined dataset hash are checked. LF checkout rules prevent platform-specific line endings changing the dataset hash.

Trade-off: this is one known repository. The results are useful for regressions, not a claim of generalization. Benchmark source answers refer to historical behavior, including bugs fixed in the current application.

## Honest release scope

Unit tests exercise HTTP success, failure and retry behavior through simulated Ollama responses. An actual local model is not installed/running in the development environment used for this change. Answer quality and inference latency are deliberately unreported. The manual grading workflow requires all fifty answers to be scored with reasons before it emits a quality result.
