import pytest
from src.retrieval.code_search import tokens, syntax_chunks, search
from src.retrieval.evidence import retrieve, bounded_context


def test_identifiers_share_tokens_with_natural_language():
    assert tokens('maxConcurrentFetches') == tokens('max_concurrent_fetches')
    assert tokens('cleaning comments') == tokens('clean comment')


def test_complete_function_and_decorator_retain_real_lines():
    raw = 'FLAG = True\n\n@decorate\ndef find_value():\n    return 7\n\ndef other():\n    return 8\n'
    chunks = syntax_chunks([{'path':'a.py', 'raw':raw}])
    function = next(c for c in chunks if 'def find_value' in c.text)
    assert (function.start_line, function.end_line) == (3, 5)
    assert function.text == '\n'.join(raw.splitlines()[2:5])
    assert all(c.text == '\n'.join(raw.splitlines()[c.start_line-1:c.end_line]) for c in chunks)


def test_methods_async_and_module_statements_are_retained():
    raw = 'import os\nclass Thing:\n    FLAG = 7\n    async def run(self):\n        return self.FLAG\nRESULT = 9\n'
    chunks = syntax_chunks([{'path':'a.py', 'raw':raw}])
    joined = '\n'.join(c.text for c in chunks)
    for text in ('import os', 'class Thing', 'FLAG = 7', 'async def run', 'RESULT = 9'):
        assert text in joined


@pytest.mark.parametrize('path,raw', [('bad.py', 'def broken(:\npass'), ('a.ts', 'function go() {}')])
def test_invalid_python_and_other_languages_fall_back(path, raw):
    chunks = syntax_chunks([{'path':path, 'raw':raw}])
    assert chunks[0].text == raw


def test_huge_function_is_bounded_and_no_code_is_executed():
    raw = 'def dangerous():\n' + '    raise RuntimeError("do not execute")\n' * 250
    chunks = syntax_chunks([{'path':'a.py', 'raw':raw}])
    assert all(c.end_line - c.start_line + 1 <= 100 for c in chunks)
    assert chunks[-1].end_line == 251
    with pytest.raises(ValueError):
        syntax_chunks([], max_lines=8)


def test_unknown_query_does_not_get_zero_overlap_results():
    assert search('unicorn', [{'path':'a.py','raw':'def add(x, y): return x + y'}]) == []
    with pytest.raises(ValueError):
        search('x', [], top_k=0)


def test_benchmark_independent_identifier_lookup():
    files = [{'path':'module.py', 'raw':'def calculateInvoiceTotal(items):\n    """Sum invoice values."""\n    return sum(items)\n'},
             {'path':'other.py', 'raw':'def drawCircle(radius):\n    return radius * radius\n'}]
    result = retrieve('How is invoice total calculated?', files, mode='code')
    assert result[0].path == 'module.py'
    assert 'return sum(items)' in result[0].text
    assert bounded_context(result, max_chars=1) == []


def test_parsed_only_fallback_does_not_claim_original_lines():
    chunks = syntax_chunks([{'path':'a.py', 'parsed':'function metadata'}])
    assert not chunks[0].raw_source
