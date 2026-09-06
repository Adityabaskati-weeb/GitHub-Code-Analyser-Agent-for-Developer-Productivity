"""Rebuild the benchmark corpus from an immutable commit of this repository."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVISION = "24d5b624e881cc2ca66a3fd2664be1b2d2d8f5c8"


def main():
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", REVISION], cwd=ROOT, text=True
    ).splitlines()
    files = []
    for path in paths:
        if not (path.startswith("src/") and path.endswith(".py") or path in {
            "run_cli.py", "langgraph_app.py", "state_schema.py", "requirements.txt",
            ".github/workflows/ci.yml"
        }):
            continue
        raw = subprocess.check_output(["git", "show", f"{REVISION}:{path}"], cwd=ROOT).decode("utf-8")
        files.append({"path": path, "raw": raw,
                      "sha256": hashlib.sha256(raw.encode()).hexdigest()})
    corpus = {"repository": "Adityabaskati-weeb/GitHub-Code-Analyser-Agent-for-Developer-Productivity",
              "revision": REVISION, "files": files}
    (ROOT / "evaluation/corpus.json").write_text(json.dumps(corpus, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Frozen {len(files)} files from {REVISION}")


if __name__ == "__main__":
    main()
