"""
src/retrieval/ranker.py
=======================
Ranks text chunks by relevance to a query.

Two modes
---------
* **lexical** (default, always available):
  TF-IDF-weighted cosine similarity using only stdlib ``math`` and
  ``collections``.

* **semantic** (optional):
  Uses ``sentence-transformers`` if it is installed; gracefully falls
  back to lexical mode if the import fails or the model cannot load.

The active mode can be forced via ``RETRIEVAL_MODE`` in the environment
("lexical" | "semantic").
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

from src.config.settings import RETRIEVAL_MODE

if TYPE_CHECKING:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Internal helpers – lexical
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    total = len(tokens) or 1
    return {t: count / total for t, count in Counter(tokens).items()}


def _idf(term: str, corpus: list[list[str]]) -> float:
    containing = sum(1 for doc in corpus if term in doc)
    return math.log((len(corpus) + 1) / (containing + 1)) + 1.0


def _tfidf_vector(tokens: list[str], corpus: list[list[str]]) -> dict[str, float]:
    tf = _tf(tokens)
    return {t: score * _idf(t, corpus) for t, score in tf.items()}


def _cosine(v1: dict[str, float], v2: dict[str, float]) -> float:
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[t] * v2[t] for t in common)
    mag1 = math.sqrt(sum(x * x for x in v1.values())) or 1.0
    mag2 = math.sqrt(sum(x * x for x in v2.values())) or 1.0
    return dot / (mag1 * mag2)


def _lexical_rank(query: str, chunks: list[str], top_k: int) -> list[str]:
    if not chunks:
        return []
    tokenised_chunks = [_tokenize(c) for c in chunks]
    corpus = tokenised_chunks
    q_tokens = _tokenize(query)
    q_vec = _tfidf_vector(q_tokens, corpus)
    scored = [
        (_cosine(q_vec, _tfidf_vector(tok, corpus)), chunk)
        for tok, chunk in zip(tokenised_chunks, chunks)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# ---------------------------------------------------------------------------
# Internal helpers – semantic (optional)
# ---------------------------------------------------------------------------

def _semantic_rank(query: str, chunks: list[str], top_k: int) -> list[str]:
    """Attempt semantic ranking; fall back to lexical on any error."""
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore

        model = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = model.encode(query, convert_to_tensor=True)
        c_embs = model.encode(chunks, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, c_embs)[0].tolist()
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]
    except Exception:  # noqa: BLE001 – graceful degradation
        return _lexical_rank(query, chunks, top_k)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_chunks(
    query: str,
    chunks: list[str],
    top_k: int = 8,
    mode: str | None = None,
) -> list[str]:
    """
    Return the *top_k* most relevant chunks for *query*.

    Parameters
    ----------
    query:  The user's natural-language question.
    chunks: Flat list of text chunks to rank.
    top_k:  Maximum number of chunks to return.
    mode:   ``"lexical"`` or ``"semantic"``.  Defaults to ``RETRIEVAL_MODE``
            from settings.

    Returns
    -------
    Ordered list of the most relevant chunks (best first).
    """
    if not chunks:
        return []
    if len(chunks) <= top_k:
        return chunks  # nothing to rank

    effective_mode = (mode or RETRIEVAL_MODE or "lexical").lower()

    if effective_mode == "semantic":
        return _semantic_rank(query, chunks, top_k)
    return _lexical_rank(query, chunks, top_k)
