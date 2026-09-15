"""Retrieval shared by the application and benchmark; retains source provenance."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from src.retrieval.ranker import rank_chunks


@dataclass(frozen=True)
class Evidence:
    path: str
    start_line: int
    end_line: int
    text: str
    raw_source: bool = True

    def render(self):
        location = f"{self.path}:L{self.start_line}-L{self.end_line}"
        if not self.raw_source:
            location = f"{self.path} (parsed text, not source line numbers)"
        return f"### {location}\n{self.text}"

    def to_dict(self):
        return asdict(self)


def build_chunks(files: list[dict], chunk_lines: int = 40, overlap: int = 8):
    if chunk_lines <= 0 or not 0 <= overlap < chunk_lines:
        raise ValueError("Require chunk_lines > 0 and 0 <= overlap < chunk_lines")
    chunks = []
    for file in files:
        if file.get("error"):
            continue
        raw_source = "raw" in file
        lines = file.get("raw", file.get("parsed", "")).splitlines()
        for start in range(0, len(lines), chunk_lines - overlap):
            end = min(start + chunk_lines, len(lines))
            text = "\n".join(line[:1000] for line in lines[start:end])
            if text.strip():
                chunks.append(Evidence(file["path"], start + 1, end, text, raw_source))
            if end == len(lines):
                break
    return chunks


def retrieve(query: str, files: list[dict], top_k: int = 8, mode: str = "lexical",
             chunk_lines: int = 40, overlap: int = 8):
    if mode in {"code", "code_compact"}:
        from src.retrieval.code_search import search
        return search(query, files, top_k)
    chunks = build_chunks(files, chunk_lines, overlap)
    by_text = {chunk.render(): chunk for chunk in chunks}
    ranked = rank_chunks(query, list(by_text), top_k=top_k, mode=mode)
    return [by_text[text] for text in ranked]


def bounded_context(chunks, max_chars: int = 18000):
    """Fit complete evidence blocks to a character budget."""
    accepted, used = [], 0
    for chunk in chunks:
        size = len(chunk.render()) + 2
        if used + size <= max_chars:
            accepted.append(chunk)
            used += size
    return accepted
