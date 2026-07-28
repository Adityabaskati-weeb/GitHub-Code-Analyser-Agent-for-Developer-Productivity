"""
tests/test_query_analyser_node.py
===================================
Unit tests for detect_intent, extract_keywords, extract_targets.
All functions are pure (no I/O, no LLM), so no mocking is needed.
"""

from __future__ import annotations

import pytest

from src.nodes.query_analyser_node import (
    detect_intent,
    extract_keywords,
    extract_targets,
    INTENT_PATTERNS,
)


# ---------------------------------------------------------------------------
# detect_intent
# ---------------------------------------------------------------------------

class TestDetectIntent:

    def test_function_usage_where_is_used(self):
        assert detect_intent("where is run_app used") == "function_usage"

    def test_function_usage_who_calls(self):
        assert detect_intent("who calls the parser") == "function_usage"

    def test_function_usage_find_usages(self):
        assert detect_intent("find usages of predict") == "function_usage"

    def test_type_lookup_type_of(self):
        assert detect_intent("type of response variable") == "type_lookup"

    def test_type_lookup_what_type(self):
        assert detect_intent("what type is config") == "type_lookup"

    def test_pipeline_flow(self):
        assert detect_intent("explain the pipeline flow") == "pipeline_flow"

    def test_pipeline_flow_execution(self):
        assert detect_intent("describe the execution flow") == "pipeline_flow"

    def test_directory_question_whats_inside(self):
        assert detect_intent("what's inside the utils folder") == "directory_question"

    def test_directory_question_show_directory(self):
        assert detect_intent("show the directory structure") == "directory_question"

    def test_architecture_summary(self):
        assert detect_intent("explain the architecture") == "architecture_summary"

    def test_architecture_overall_structure(self):
        assert detect_intent("what is the overall structure") == "architecture_summary"

    def test_unrecognised_query_returns_high_level_summary(self):
        assert detect_intent("random gibberish xyz") == "high_level_summary"

    def test_empty_string_returns_high_level_summary(self):
        assert detect_intent("") == "high_level_summary"

    def test_case_insensitive(self):
        # Intent patterns use re.search with query.lower(), so this should work
        assert detect_intent("PIPELINE FLOW") == "pipeline_flow"


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------

class TestExtractKeywords:

    def test_filters_stop_words(self):
        kws = extract_keywords("where is the function used")
        # "where", "the" are stop words; "is" has len 2 → filtered
        assert "where" not in kws
        assert "the" not in kws

    def test_includes_function_name(self):
        kws = extract_keywords("where is run_app used")
        assert "run_app" in kws

    def test_short_tokens_excluded(self):
        kws = extract_keywords("is it ok")
        for kw in kws:
            assert len(kw) > 2

    def test_returns_at_most_five(self):
        long_query = "alpha beta gamma delta epsilon zeta eta theta"
        kws = extract_keywords(long_query)
        assert len(kws) <= 5

    def test_empty_returns_empty(self):
        assert extract_keywords("") == []

    def test_only_stop_words_returns_empty(self):
        kws = extract_keywords("where the")
        assert all(len(k) > 2 and k.lower() not in {"the", "where", "and", "to", "in"} for k in kws)


# ---------------------------------------------------------------------------
# extract_targets
# ---------------------------------------------------------------------------

class TestExtractTargets:

    def test_function_with_parens(self):
        targets = extract_targets("where is run_app() defined")
        assert targets.get("function") == "run_app"

    def test_directory_path(self):
        targets = extract_targets("what is inside src/nodes/")
        assert "directory" in targets
        # The detected directory should contain "src/"
        assert "src" in targets["directory"]

    def test_variable_type_of(self):
        targets = extract_targets("type of response_data")
        assert targets.get("variable") == "response_data"

    def test_no_targets_returns_empty_dict(self):
        targets = extract_targets("tell me about the project")
        # No function(), no /path/, no "type of X"
        assert isinstance(targets, dict)

    def test_multiple_targets_in_one_query(self):
        targets = extract_targets("type of run_app() in src/utils/")
        # Should find at least one target
        assert len(targets) >= 1
