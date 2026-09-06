"""Read a local checkout for inference without GitHub network access."""
from pathlib import Path

from src.config.settings import MAX_SIZE_KB

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".cache",
             "reports", ".pytest_cache", "dist", "build"}
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml",
                   ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java"}


def read_local_repo(root: str, max_files: int = 2000) -> list[dict]:
    import os
    base = Path(root).resolve()
    if not base.is_dir():
        raise ValueError(f"Local repository directory does not exist: {base}")
    files = []
    for directory, dirs, names in os.walk(base, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                         and (not d.startswith(".") or d == ".github")
                         and not (Path(directory) / d).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            if path.is_symlink() or name.startswith(".env") or path.suffix not in TEXT_EXTENSIONS:
                continue
            if path.stat().st_size > MAX_SIZE_KB * 1024:
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except (UnicodeError, OSError):
                continue
            if "\x00" in raw:
                continue
            if len(files) >= max_files:
                raise ValueError(f"Local repository exceeds {max_files} eligible files; use a smaller directory.")
            files.append({"path": path.relative_to(base).as_posix(), "raw": raw,
                          "parsed": raw, "ext": path.suffix})
    return files
