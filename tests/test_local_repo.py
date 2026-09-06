from pathlib import Path
import subprocess
import sys

import pytest

from src.local_repo import read_local_repo


def test_local_reader_excludes_env_dependencies_and_binary(tmp_path):
    (tmp_path / "a.py").write_text("value = 42", encoding="utf-8")
    (tmp_path / ".env.json").write_text('{"key":"secret"}')
    (tmp_path / "binary.py").write_bytes(b"\x00\x01")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "vendor.py").write_text("vendor")
    assert [f["path"] for f in read_local_repo(str(tmp_path))] == ["a.py"]


def test_local_reader_enforces_limit(tmp_path):
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("b")
    with pytest.raises(ValueError, match="exceeds"):
        read_local_repo(str(tmp_path), max_files=1)


def test_missing_directory_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        read_local_repo(str(tmp_path / "missing"))


def test_cli_retrieval_only_runs_without_model(tmp_path):
    (tmp_path / "answer.py").write_text("def meaning():\n    return 42  # Unicode: → café\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "run_cli.py", str(tmp_path), "--local",
                             "--retrieval-only", "--no-report", "-q", "meaning"],
                            cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert "answer.py:L1-L2" in result.stdout
    assert "return 42" in result.stdout
    assert "→ café" in result.stdout
