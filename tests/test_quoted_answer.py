import json
import pytest
from src.retrieval.evidence import Evidence
from src.retrieval.quoted_answer import render_quoted

CATALOG = {'S1':Evidence('a.py', 1, 3, 'def answer():\n    return ["done"]')}


def result(quote='return ["done"]', **changes):
    data = {'answer':'Returns a list containing done.', 'insufficient_evidence':False,
            'evidence':[{'source_id':'S1','quote':quote}]}
    data.update(changes)
    return render_quoted(json.dumps(data), CATALOG)


def test_exact_quote_maps_to_real_source():
    text, abstained, check = result()
    assert not abstained
    assert 'a.py:L1-L3' in text
    assert check['status'] == 'valid'


@pytest.mark.parametrize('changes', [{'evidence':[]}, {'insufficient_evidence':True},
    {'evidence':[{'source_id':'S999','quote':'return ["done"]'}]}])
def test_missing_or_unknown_evidence_abstains(changes):
    text, abstained, _ = result(**changes)
    assert abstained
    assert 'Returns a list' not in text


def test_fabricated_or_too_short_quote_abstains():
    for quote in ('return 12345', 'done'):
        _, abstained, check = result(quote)
        assert abstained and check['status'] == 'invalid_quote'


def test_whitespace_only_changes_are_accepted():
    assert not result('def answer(): return ["done"]')[1]


def test_matching_quote_does_not_prove_answer_truth():
    # Deliberately documents the boundary: quote checking is not an entailment judge.
    _, abstained, check = result(answer='Invented interpretation.')
    assert not abstained and check['status'] == 'valid'


def test_quote_is_relocated_only_when_exact_and_unique():
    catalog = {**CATALOG, 'S2':Evidence('b.py', 5, 6, 'return 123456')}
    raw = json.dumps({'answer':'Returns 123456.', 'insufficient_evidence':False,
                      'evidence':[{'source_id':'S1','quote':'return 123456'}]})
    text, abstained, check = render_quoted(raw, catalog)
    assert not abstained and '[b.py:L5-L6]' in text
    assert check['source_id_corrections'] == [{'from':'S1','to':'S2'}]
    catalog['S3'] = Evidence('c.py', 1, 1, 'return 123456')
    assert render_quoted(raw, catalog)[1]


def test_malformed_structure_fails_explicitly():
    for raw in ('{}', '{"answer":"truncated', '[]'):
        answer, abstained, check = render_quoted(raw, CATALOG)
        assert abstained and check['status'] == 'malformed_response'
        assert 'no answer was accepted' in answer


def test_empty_answer_is_not_published():
    assert result(answer='')[1]


def test_answer_node_uses_quote_protocol_for_code_mode():
    import asyncio
    from unittest.mock import AsyncMock
    from langchain_core.messages import AIMessage, HumanMessage
    from src.nodes.summarize_repo_node import summarize_repo_node
    llm = AsyncMock()
    llm.supports_source_schema = True
    llm.ainvoke.return_value = AIMessage(content=json.dumps({'answer':'Returns 42.',
        'evidence':[{'source_id':'S1','quote':'return 42'}], 'insufficient_evidence':False}))
    state = {'parsed_files':[{'path':'a.py','raw':'def answer(): return 42'}],
             'messages':[HumanMessage(content='What does answer return?')], 'llm':llm,
             'retrieval_mode':'code'}
    response = asyncio.run(summarize_repo_node(state))
    assert response['metrics']['quote_check']['status'] == 'valid'
    assert response['metrics']['answer_protocol'] == 'quoted_v2'
    assert 'evidence' in llm.ainvoke.call_args.kwargs['response_format']['properties']
