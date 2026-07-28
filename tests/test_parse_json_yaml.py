"""
tests/test_parse_json_yaml.py
==============================
Unit tests for src.tools.parse_json_yaml.parse_json_yaml
"""

from __future__ import annotations

import json
import pytest

from src.tools.parse_json_yaml import parse_json_yaml


class TestParseJsonYaml:

    # ------------------------------------------------------------------
    # JSON inputs
    # ------------------------------------------------------------------

    def test_valid_json_object(self):
        raw = '{"name": "alice", "age": 30}'
        result = parse_json_yaml(raw)
        # Should be pretty-printed JSON
        parsed_back = json.loads(result)
        assert parsed_back["name"] == "alice"
        assert parsed_back["age"] == 30

    def test_valid_json_array(self):
        raw = '[1, 2, 3]'
        result = parse_json_yaml(raw)
        assert json.loads(result) == [1, 2, 3]

    def test_json_output_is_indented(self):
        raw = '{"a": 1}'
        result = parse_json_yaml(raw)
        assert "\n" in result  # pretty-printed → has newlines

    # ------------------------------------------------------------------
    # YAML inputs
    # ------------------------------------------------------------------

    def test_valid_yaml_mapping(self):
        raw = "name: bob\nage: 25\n"
        result = parse_json_yaml(raw)
        assert "bob" in result
        assert "age" in result

    def test_valid_yaml_list(self):
        raw = "- item1\n- item2\n"
        result = parse_json_yaml(raw)
        assert "item1" in result

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------

    def test_output_never_exceeds_5000_chars(self):
        # Construct a large JSON blob
        big = {str(i): "x" * 100 for i in range(200)}
        raw = json.dumps(big)
        result = parse_json_yaml(raw)
        assert len(result) <= 5000

    def test_large_yaml_truncated(self):
        raw = "\n".join(f"key{i}: {'v' * 50}" for i in range(200))
        result = parse_json_yaml(raw)
        assert len(result) <= 5000

    # ------------------------------------------------------------------
    # Fallback for invalid inputs
    # ------------------------------------------------------------------

    def test_invalid_input_returns_raw_head(self):
        garbage = "not json or yaml: :::{{ bad"
        result = parse_json_yaml(garbage)
        # Should return something ≤ 5000 chars
        assert len(result) <= 5000
        assert isinstance(result, str)

    def test_empty_string_does_not_raise(self):
        result = parse_json_yaml("")
        assert isinstance(result, str)
