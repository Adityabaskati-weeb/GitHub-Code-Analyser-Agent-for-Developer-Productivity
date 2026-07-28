# GitHub Code-Analyser Agent for Developer Productivity

A LangGraph-powered CLI agent that indexes a GitHub repository, selects relevant files, parses source content, and answers developer-productivity questions about architecture, pipeline flow, functions, directories, and implementation details.

## Features

- Fetches GitHub repository metadata with default-branch detection.
- Filters and selects relevant files for a user question.
- Parses Python, Markdown, JSON, YAML, and notebooks.
- Builds a high-level repository context with Gemini.
- Answers follow-up questions through a LangGraph QA workflow.
- Supports interactive and one-shot CLI usage.
- Includes a local smoke test that runs without network or LLM calls.

## Project Structure

```text
run_cli.py                  # CLI entrypoint
langgraph_app.py            # LangGraph workflow definitions
state_schema.py             # Shared graph state
src/
  config/settings.py        # Environment and model settings
  github_repo_parser.py     # GitHub repository tree parser
  nodes/                    # LangGraph node implementations
  tools/                    # File parsers
  utils/                    # Helper utilities
```

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

Create a `.env` file:

```bash
cp .env.example .env
```

Set your Gemini API key:

```env
GOOGLE_API_KEY=your_key_here
GOOGLE_MODEL=gemini-2.5-flash
```

`GITHUB_TOKEN` is optional for public repositories, but recommended to avoid GitHub rate limits.

## Run

Smoke test without network or API calls:

```bash
python run_cli.py --smoke-test
```

Interactive mode:

```bash
python run_cli.py https://github.com/owner/repo
```

Ask one question and exit:

```bash
python run_cli.py https://github.com/owner/repo --question "Explain the architecture"
```

## Environment

| Variable | Purpose |
| --- | --- |
| `GOOGLE_API_KEY` | Required Gemini API key. |
| `GEMINI_API_KEY` | Alternative Gemini API key variable. |
| `GOOGLE_MODEL` | Gemini model name. Defaults to `gemini-2.5-flash`. |
| `GITHUB_TOKEN` | Optional GitHub token for higher API limits. |

## Verification

```bash
python -m py_compile run_cli.py langgraph_app.py state_schema.py
python run_cli.py --smoke-test
```
