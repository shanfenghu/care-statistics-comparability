"""Boundary tests for exact inputs and population-count comparisons."""
import tempfile
import unittest
from pathlib import Path

from acquire import check, safe
from check_results import compare
from export_tables import count


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


if __name__ == '__main__':
    unittest.main()
