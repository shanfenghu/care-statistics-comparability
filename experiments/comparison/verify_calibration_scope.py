"""Numerical and policy checks for verify calibration scope."""
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

from comparison_panel import digest

VENUE = Path(__file__).resolve().parents[2]


def main():
    source = VENUE / 'results/calibration-scope/summary.json'
    result = json.loads(source.read_text())
    folder = source.parent
    for field, path in [('protocol_sha256', VENUE/'experiments/comparison/config/calibration-method.txt'),
                        ('script_sha256', VENUE/'experiments/comparison/calibration_scope.py'),
                        ('policies_sha256', folder/'policies.json'),
                        ('source_summary_sha256', VENUE/'results/comparison-robustness/summary.json')]:
        assert digest(path) == result[field]
    for row in result['source_cases']:
        path = VENUE/row['path']
        assert digest(path) == row['sha256']
        case = json.loads(path.read_text())
        for saved in case['methods'].values():
            assert all(x > 0 for row in saved['test_prediction'] for x in row)
    records = json.loads((folder/'policies.json').read_text())
    assert len(records) == 3240 and len(result['source_cases']) == 35
    sums = defaultdict(lambda: defaultdict(int))
    integer_fields = ('queries', 'resolved', 'answered_all', 'answered', 'wrong',
                      'added', 'added_wrong', 'removed', 'removed_wrong', 'rows',
                      'covered_rows', 'covered_vectors', 'infinite')
    index = {(r['case'], r['method'], r['alpha'], r['scope'], r['family']): r for r in records}
    assert len(index) == len(records)
    for row in records:
        base = index[(row['case'], row['method'], row['alpha'], 'vector', row['family'])]
        assert row['resolved'] == base['resolved']
        assert row['answered'] == base['answered'] + row['added'] - row['removed']
        assert row['wrong'] == base['wrong'] + row['added_wrong'] - row['removed_wrong']
        assert 0 <= row['wrong'] <= row['answered'] <= row['resolved'] <= row['queries']
        assert 0 <= row['covered_vectors'] <= row['covered_rows'] <= row['rows']
        if row['scope'] == 'specified-family':
            assert row['removed'] == 0 and row['removed_wrong'] == 0
        for f in (str(row['family']), 'all'):
            key = (row['group'], row['method'], row['alpha'], row['scope'], f)
            for field in integer_fields:
                sums[key][field] += row[field]
    for row in result['summaries']:
        key = (row['group'], row['method'], row['alpha'], row['scope'], row['family'])
        for field in integer_fields:
            assert row[field] == sums[key][field], (key, field)
        assert row['coverage'] == row['answered']/row['resolved']
        assert row['risk'] == (row['wrong']/row['answered'] if row['answered'] else None)
    test = subprocess.run([sys.executable,
                           str(VENUE/'tests/comparison/test_calibration_scope.py')],
                          capture_output=True, text=True)
    if test.returncode:
        raise RuntimeError(test.stdout + test.stderr)
    verification = dict(summary_sha256=digest(source), verifier_sha256=digest(__file__),
                        tests_sha256=digest(VENUE/'tests/comparison/test_calibration_scope.py'),
                        constructed_tests=8, cases=35, policy_records=len(records),
                        aggregate_rows=len(result['summaries']),
                        **result['verification'],
                        all_added_removed_error_counts_reconciled=True,
                        all_original_input_case_hashes_unchanged=True,
                        positive_test_predictions_verified=True)
    (folder/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
