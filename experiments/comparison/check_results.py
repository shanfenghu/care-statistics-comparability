"""Compare all selected scientific payloads, with exact integers and bounded float error."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = ROOT / 'experiments/comparison/config/expected.json'
COUNT_FIELDS = {'queries','resolved','answered_all','answered','wrong','added',
                'added_wrong','removed','removed_wrong','rows','covered_rows',
                'covered_vectors','infinite','resolved_queries','answers_on_resolved',
                'wrong_on_resolved'}


def compare(expected, actual, path='root'):
    """Return checked leaves; counts/labels are exact, floats use 1e-8/1e-10."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(expected) <= set(actual), path
        return sum(compare(value, actual[key], path + '/' + key)
                   for key, value in expected.items())
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), path
        return sum(compare(x, y, path + '/' + str(i)) for i, (x, y) in enumerate(zip(expected, actual)))
    if isinstance(expected, float):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), path
        # Some retained unweighted totals were serialised as integral floats.
        # Their storage type must not turn discrete error counts into toleranced rates.
        if path.rsplit('/',1)[-1] in COUNT_FIELDS:
            assert expected.is_integer() and float(actual).is_integer() and expected==actual, (path, expected, actual)
        else:
            assert math.isclose(expected, actual, rel_tol=1e-8, abs_tol=1e-10), (path, expected, actual)
    else:
        assert type(expected) is type(actual) and expected == actual, (path, expected, actual)
    return 1


def main():
    totals = {}
    for group, expected in json.loads(EXPECTED.read_text()).items():
        actual = json.loads((ROOT / 'results' / group / 'summary.json').read_text())
        totals[group] = compare(expected, actual, group)
    record = dict(checked_leaves=sum(totals.values()), groups=totals,
                  integer_comparison='exact', relative_tolerance=1e-8, absolute_tolerance=1e-10,
                  scope='Selected fixed-data scientific payloads; not independent validation.')
    (ROOT / 'records/result-check.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
