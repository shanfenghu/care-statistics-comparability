"""Boundary tests for exact inputs and population-count comparisons."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import acquire
from acquire import check, safe
from check_results import compare
from export_tables import count
import reproduce


class InputBoundaryTests(unittest.TestCase):
    def test_escaping_and_symlink_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['/etc/passwd', '../outside', 'a/../../outside']:
                with self.assertRaises(ValueError):
                    safe(root,name)
            (root/'link').symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                safe(root,'link/input.csv')
            self.assertEqual(safe(root,'data/file.csv'),root/'data/file.csv')

    def test_changed_bytes_and_sizes_fail(self):
        import hashlib
        item = dict(path='data/file.csv',bytes=3,sha256=hashlib.sha256(b'abc').hexdigest())
        check(b'abc',item)
        for data in [b'xyz',b'abcd']:
            with self.assertRaises(ValueError):check(data,item)

    def test_provider_download_validates_all_inputs_before_writing(self):
        import hashlib
        import io
        import json
        payloads = [b'abc', b'def']
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            items = [dict(path='data/raw/' + str(i) + '.csv', bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest(),
                          url='https://example.invalid/' + str(i))
                     for i, data in enumerate(payloads)]
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(dict(files=items)))
            for valid in (False, True):
                responses = [io.BytesIO(payloads[0]),
                             io.BytesIO(payloads[1] if valid else b'bad')]
                with patch.object(acquire, 'ROOT', root), \
                        patch.object(acquire, 'MANIFEST', manifest), \
                        patch('sys.argv', ['acquire', '--download']), \
                        patch.object(acquire.urllib.request, 'urlopen', side_effect=responses) as http:
                    if valid:
                        acquire.main()
                    else:
                        with self.assertRaisesRegex(ValueError, 'Input version differs'):
                            acquire.main()
                        self.assertFalse((root / 'data').exists())
                    self.assertEqual([c.args[0] for c in http.call_args_list],
                                     [item['url'] for item in items])
            for item, data in zip(items, payloads):
                self.assertEqual((root / item['path']).read_bytes(), data)

    def test_result_counts_are_exact_and_floats_have_declared_tolerance(self):
        self.assertEqual(compare({'count':3,'rate':.1},{'count':3,'rate':.1+1e-12}),2)
        for actual in [{'count':4,'rate':.1}, {'count':3.,'rate':.1}, {'count':3,'rate':.2}]:
            with self.assertRaises(AssertionError):compare({'count':3,'rate':.1},actual)

    def test_integral_float_counts_remain_exact(self):
        self.assertEqual(compare({'wrong':1.0},{'wrong':1.0}),1)
        with self.assertRaises(AssertionError):compare({'wrong':1.0},{'wrong':1.0+1e-12})
        self.assertEqual(count(81.0),81)
        for bad in [True, -1, 1.5, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):count(bad)

    def test_partial_outputs_are_preserved_before_any_stage(self):
        names = ['results/' + name + '/summary.json' for name in (
            'comparison-panel', 'comparison-robustness', 'comparison-ranking-stable',
            'comparison-ranking-boundary', 'comparison-mechanism', 'calibration-scope')]
        names += ['records/steps.json', 'figures/reuse-example.pdf',
                  'figures/reuse-fit.png', 'tables/geographic.csv', 'tables/metrics.json']
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                path = root / name
                path.parent.mkdir(parents=True)
                path.write_bytes(b'earlier evidence')
                before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
                with patch.object(reproduce, 'ROOT', root), patch('sys.argv', ['reproduce']), \
                        patch.object(reproduce.subprocess, 'run') as child:
                    with self.assertRaises(FileExistsError):
                        reproduce.main()
                    child.assert_not_called()
                self.assertEqual(path.read_bytes(), b'earlier evidence')
                self.assertEqual(sorted(str(p.relative_to(root)) for p in root.rglob('*')), before)

    def test_output_symlinks_are_refused(self):
        for name in ('results', 'records', 'figures', 'tables',
                     'results/comparison-robustness', 'figures/reuse-fit.pdf',
                     'tables/metrics.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(root / 'missing')
                with self.assertRaises(FileExistsError):
                    reproduce.require_fresh_outputs(root)

    def test_unrelated_outputs_and_empty_parents_are_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('results/other-evaluation', 'records', 'figures', 'tables'):
                (root / name).mkdir(parents=True)
            (root / 'figures/other.pdf').write_bytes(b'earlier evidence')
            reproduce.require_fresh_outputs(root)


if __name__ == '__main__':
    unittest.main()
