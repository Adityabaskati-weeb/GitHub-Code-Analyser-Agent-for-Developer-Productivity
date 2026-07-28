"""
src/retrieval/chunker.py
========================
Splits long text into overlapping word-boundary chunks.
No external dependencies – only stdlib.
"""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[str]:
    """
    Split *text* into overlapping word-level chunks.

    Parameters
    ----------
    text:       The source text to split.
    chunk_size: Maximum number of words per chunk.
    overlap:    Number of words shared between consecutive chunks.

    Returns
    -------
    A list of string chunks.  Returns ``[""]`` if *text* is empty.
    """
    if not text or not text.strip():
        return [""]

    words = text.split()
    if not words:
        return [""]

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks if chunks else [""]
