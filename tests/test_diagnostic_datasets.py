import json
from evaluation.run import HERE, load_dataset
from evaluation.oracle import oracle_chunks
from src.retrieval.code_search import syntax_chunks


def test_oracle_evidence_contains_each_baseline_failure_anchor():
    corpus, questions = load_dataset()
    baseline = json.loads((HERE / 'live_answers.json').read_text(encoding='utf-8'))
    failures = {r['id'] for r in baseline['rows'] if r['score']==0}
    for q in questions:
        if q['id'] in failures:
            chunks = oracle_chunks(q, corpus['files'])
            assert len(chunks) == 1
            assert chunks[0].path == q['path'] and q['anchor'] in chunks[0].text


def test_synthetic_cases_include_answerable_and_unanswerable_queries():
    data = json.loads((HERE / 'challenges.json').read_text(encoding='utf-8'))
    assert len({q['id'] for q in data['questions']}) == 8
    assert sum(q['answerable'] for q in data['questions']) == 4
    source = '\n'.join(c.text for c in syntax_chunks(data['files']))
    assert 'LEASE_SECONDS = 45' in source
    assert 'return {}' in source and 'return []' in source
