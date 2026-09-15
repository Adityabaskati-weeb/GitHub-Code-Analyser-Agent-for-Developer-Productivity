"""Compare complete, declared source reviews; never equate citation validity with truth."""
import argparse
import json
from pathlib import Path

from evaluation.answers import score


def metrics(report):
    result = score(report)
    rows = report['rows']
    result.update(
        abstentions=sum(bool(r['metrics']['abstained']) for r in rows),
        incorrect_delivered_answers=sum(r['score'] == 0 and not r['metrics']['abstained'] for r in rows),
        partial_answers=sum(r['score'] == 1 for r in rows),
    )
    return result


def compare(baseline, candidate):
    before, after = metrics(baseline), metrics(candidate)
    failures = []
    for key in ('fully_correct_fraction', 'mean_score_out_of_2'):
        if after[key] < before[key]:
            failures.append(f'{key} regressed: {before[key]} -> {after[key]}')
    if after['incorrect_delivered_answers'] > before['incorrect_delivered_answers']:
        failures.append('More incorrect answers were delivered')
    return {'scope':'Paired authored-set reviews, not independent accuracy or a causal ablation.',
            'baseline':before, 'candidate':after,
            'regression_gate_passed':not failures, 'failures':failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--check', action='store_true', help='Exit 1 if reviewed answer quality regressed')
    args = parser.parse_args()
    result = compare(*(json.loads(p.read_text(encoding='utf-8')) for p in (args.baseline, args.candidate)))
    print(json.dumps(result, indent=2))
    return int(args.check and not result['regression_gate_passed'])


if __name__ == '__main__':
    raise SystemExit(main())
