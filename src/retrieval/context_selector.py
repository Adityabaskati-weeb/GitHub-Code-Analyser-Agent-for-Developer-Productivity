"""Deterministic whole-excerpt selection for the compact code candidate."""
from __future__ import annotations

from src.retrieval.evidence import Evidence


COMPACT_CONTEXT_VERSION = "compact_v1"
COMPACT_CONTEXT_BUDGET = 8000
COMPACT_MAX_EXCERPTS = 4


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _near_contained(candidate: Evidence, selected: list[Evidence]) -> bool:
    """Drop exact/contained/mostly-overlapping spans from the same file."""
    candidate_text = _normalized(candidate.text)
    for prior in selected:
        if prior.path != candidate.path:
            continue
        if candidate_text and candidate_text == _normalized(prior.text):
            return True
        candidate_start, candidate_end = candidate.start_line, candidate.end_line
        prior_start, prior_end = prior.start_line, prior.end_line
        overlap = max(0, min(candidate_end, prior_end) - max(candidate_start, prior_start) + 1)
        candidate_lines = max(1, candidate_end - candidate_start + 1)
        prior_lines = max(1, prior_end - prior_start + 1)
        if (candidate_start >= prior_start and candidate_end <= prior_end
                or prior_start >= candidate_start and prior_end <= candidate_end
                or overlap / min(candidate_lines, prior_lines) >= 0.8):
            return True
    return False


def rendered_context_chars(chunks) -> int:
    """Character count of the exact double-newline rendering used in prompts."""
    return sum(len(chunk.render()) for chunk in chunks) + max(0, len(chunks) - 1) * 2


def select_compact_context(chunks, max_excerpts=COMPACT_MAX_EXCERPTS,
                           max_chars=COMPACT_CONTEXT_BUDGET):
    """Select ranked, complete evidence blocks within the compact budget.

    Input order is relevance order and is preserved.  Blocks are never cut or
    rewritten: an excerpt that cannot fit is skipped, leaving its original
    source line mapping truthful.  This intentionally favors useful excerpts
    from one file when they are ranked above distractors in other files.
    """
    if max_excerpts <= 0 or max_chars <= 0:
        raise ValueError("Compact context limits must be positive")
    selected: list[Evidence] = []
    used = 0
    for chunk in chunks:
        if not isinstance(chunk, Evidence):
            raise TypeError("Compact context expects Evidence blocks")
        if _near_contained(chunk, selected):
            continue
        size = len(chunk.render()) + (2 if selected else 0)
        if used + size > max_chars:
            continue
        selected.append(chunk)
        used += size
        if len(selected) >= max_excerpts:
            break
    return selected


def context_selection_metadata(chunks, candidates, max_excerpts=COMPACT_MAX_EXCERPTS,
                               max_chars=COMPACT_CONTEXT_BUDGET):
    """Return auditable selection facts for answer/evaluation reports."""
    used = rendered_context_chars(chunks)
    return {
        "context_selector": COMPACT_CONTEXT_VERSION,
        "context_budget_chars": max_chars,
        "max_context_excerpts": max_excerpts,
        "candidate_chunks": len(candidates),
        "selected_chunks": len(chunks),
        "selected_context_chars": used,
        "selected_sources": [chunk.to_dict() for chunk in chunks],
    }
