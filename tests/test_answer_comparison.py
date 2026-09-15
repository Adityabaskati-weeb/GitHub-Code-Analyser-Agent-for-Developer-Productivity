import copy
import json
import pytest
from evaluation.run import HERE
from evaluation.compare_answers import compare


def baseline():
    return json.loads((HERE / 'live_answers.json').read_text(encoding='utf-8'))


def test_identical_review_passes():
    report = baseline()
    assert compare(report, report)['regression_gate_passed']


def test_abstaining_on_everything_is_not_an_improvement():
    before = baseline()
    after = copy.deepcopy(before)
    for row in after['rows']:
        row.update(score=0, reviewer_reason='False abstention on available evidence.')
        row['metrics']['abstained'] = True
    result = compare(before, after)
    assert result['candidate']['incorrect_delivered_answers'] == 0
    assert result['candidate']['abstentions'] == 50
    assert not result['regression_gate_passed']


def test_partial_run_cannot_pass_regression_gate():
    before = baseline()
    after = copy.deepcopy(before)
    after['rows'].pop()
    with pytest.raises(ValueError, match='50 unique'):
        compare(before, after)
