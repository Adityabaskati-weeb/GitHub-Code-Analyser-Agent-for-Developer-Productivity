"""
tests/test_parse_python.py
===========================
Unit tests for src.tools.parse_python.parse_python
"""

from __future__ import annotations

from src.tools.parse_python import parse_python, _clean_raw_python


class TestParsePython:

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def test_simple_import_extracted(self):
        result = parse_python("import os\n")
        assert "os" in result

    def test_from_import_extracted(self):
        result = parse_python("from pathlib import Path\n")
        assert "pathlib" in result

    def test_multiple_imports(self):
        code = "import os\nimport sys\nfrom typing import List\n"
        result = parse_python(code)
        assert "os" in result
        assert "sys" in result
        assert "typing" in result

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------

    def test_class_name_extracted(self):
        code = "class MyModel:\n    pass\n"
        result = parse_python(code)
        assert "MyModel" in result

    def test_class_docstring_extracted(self):
        code = 'class Foo:\n    """Foo does bar."""\n    pass\n'
        result = parse_python(code)
        assert "Foo does bar" in result

    # ------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------

    def test_function_name_extracted(self):
        code = "def run_app():\n    return True\n"
        result = parse_python(code)
        assert "run_app" in result

    def test_function_docstring_extracted(self):
        code = 'def greet():\n    """Says hello."""\n    return "hi"\n'
        result = parse_python(code)
        assert "Says hello" in result

    def test_async_function_extracted(self):
        code = "async def fetch():\n    pass\n"
        result = parse_python(code)
        assert "fetch" in result

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_string_returns_string(self):
        result = parse_python("")
        assert isinstance(result, str)

    def test_bad_syntax_falls_back_to_raw(self):
        bad_code = "def broken(\n    # unclosed"
        result = parse_python(bad_code)
        # Should not raise; result is a non-empty string
        assert isinstance(result, str)
        assert len(result) > 0

    def test_comment_only_file_strips_comments(self):
        code = "# This is a comment\n# Another comment\n"
        result = parse_python(code)
        # AST won't extract anything meaningful → falls back to _clean_raw_python
        # which strips comment lines → result may be empty or whitespace
        assert isinstance(result, str)


class TestCleanRawPython:

    def test_removes_comment_lines(self):
        code = "# comment\nvalue = 1\n"
        result = _clean_raw_python(code)
        assert "# comment" not in result
        assert "value = 1" in result

    def test_inline_comments_preserved(self):
        """Only full-line comments (stripped start with '#') are removed."""
        code = "x = 1  # inline\n"
        result = _clean_raw_python(code)
        # The line doesn't start with #, so it should be kept
        assert "x = 1" in result

    def test_empty_input(self):
        assert _clean_raw_python("") == ""
