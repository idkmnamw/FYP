"""Small regression checks: no notebook execution, source downloads or model fitting."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    """Load only named pure functions; do not execute module/notebook setup."""
    if path.suffix == '.ipynb':
        cells = json.loads(path.read_text())['cells']
        sources = [''.join(c['source']) for c in cells if c['cell_type'] == 'code']
    else:
        sources = [path.read_text()]
    nodes = [node for source in sources for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if sorted(n.name for n in nodes) != sorted(names):
        raise AssertionError('Missing or duplicate calculation')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = definitions(ROOT / 'pricecatcher_config.py',
            ['installed_package_version', 'surge_mask'],
            {'np': np, 'LABEL_TOLERANCE_PCT': 1e-10})
        spec = importlib.util.spec_from_file_location('fyp_prepare', ROOT / 'prepare.py')
        cls.prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.prepare)
        tree = ast.parse((ROOT / 'pricecatcher_config.py').read_text())
        mapping = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'RESEARCH_NOTEBOOK_NAMES' for t in n.targets))
        cls.paths = definitions(ROOT / 'pricecatcher_config.py', ['notebook_path', 'resolve_input'],
            {'Path': Path, '__file__': str(ROOT / 'pricecatcher_config.py'),
             'RESEARCH_NOTEBOOK_NAMES': ast.literal_eval(mapping),
             'PROJECT_ROOT': ROOT / 'data/test-run', 'DATA_HOME': ROOT / 'data'})

    def test_renumbered_research_and_old_references_resolve(self):
        for old, new in self.paths['RESEARCH_NOTEBOOK_NAMES'].items():
            expected = ROOT / 'research' / new
            self.assertTrue(expected.is_file())
            self.assertEqual(self.paths['notebook_path'](old), expected)
            self.assertEqual(self.paths['notebook_path'](new), expected)
            self.assertEqual(self.paths['resolve_input']('repo::research/' + old), expected)
        self.assertEqual(self.paths['notebook_path']('15_relative_price_features.ipynb'),
                         ROOT / 'research/06_relative_price_features.ipynb')

    def test_research_renaming_preserves_saved_output_paths(self):
        reference = 'run::research/17_larger_optuna_budget/results'
        self.assertEqual(self.paths['resolve_input'](reference),
                         ROOT / 'data/test-run/research/17_larger_optuna_budget/results')

    def test_renamed_reference_resolution_rejects_root_escape(self):
        for reference in ['repo::../outside', 'run::../outside']:
            with self.assertRaises(ValueError):
                self.paths['resolve_input'](reference)

    def test_module_record_is_not_distribution_metadata(self):
        with patch.dict(sys.modules, {'xgboost': SimpleNamespace(__version__='3.4.1')}):
            with patch('importlib.metadata.version', side_effect=AssertionError('not a package')):
                self.assertEqual(self.functions['installed_package_version']('xgboost_module'), '3.4.1')

    def test_normal_package_versions_still_use_metadata(self):
        with patch('importlib.metadata.version', return_value='2.2.0') as version:
            self.assertEqual(self.functions['installed_package_version']('pandas'), '2.2.0')
            version.assert_called_once_with('pandas')

    def test_missing_module_is_reported_as_missing_package(self):
        from importlib import metadata
        with patch.dict(sys.modules, {'xgboost': None}):
            with self.assertRaises(metadata.PackageNotFoundError):
                self.functions['installed_package_version']('xgboost_module')

    def test_export_accepts_existing_xgboost_record(self):
        # Notebook 08 records both the module version and platform distribution.
        from importlib import metadata
        xgb_version = self.functions['installed_package_version']('xgboost_module')
        distribution = 'xgboost-cpu' if platform.system() == 'Linux' else 'xgboost'
        record = {'package_versions': {'xgboost_module': xgb_version,
                   distribution: metadata.version(distribution)}}
        with tempfile.TemporaryDirectory() as d:
            cfg = SimpleNamespace(PROJECT_ROOT=Path(d),
                installed_package_version=self.functions['installed_package_version'])
            with patch.dict(sys.modules, {'pricecatcher_config': cfg}), \
                 patch.object(self.prepare, 'saved_environment_records', return_value=[record]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.prepare.export_environment()
            exported = (Path(d) / 'environment_requirements.txt').read_text()
            self.assertNotIn('xgboost_module', exported)
            self.assertIn(f'xgboost-cpu=={xgb_version}', exported)

    def test_restore_message_uses_real_distribution(self):
        record = {'package_versions': {'xgboost_module': '0.0.0'}}
        cfg = SimpleNamespace(installed_package_version=self.functions['installed_package_version'])
        with patch.dict(sys.modules, {'pricecatcher_config': cfg}), \
             patch.object(self.prepare, 'saved_environment_records', return_value=[record]), \
             patch('platform.system', return_value='Linux'):
            with self.assertRaises(ValueError) as raised:
                self.prepare.export_environment()
        command = str(raised.exception).split('Restore all listed packages together in this environment:\n')[1]
        self.assertIn('xgboost-cpu==0.0.0', command)
        self.assertNotIn('xgboost_module==', command)

    def test_checkpoint_check_accepts_module_record_before_provenance_checks(self):
        record = {'package_versions': {'xgboost_module':
                  self.functions['installed_package_version']('xgboost_module')}}
        with tempfile.TemporaryDirectory() as d:
            pin = Path(d) / '11_deployment' / 'deployment_selection.json'
            pin.parent.mkdir(); pin.write_text('{}')
            cfg = SimpleNamespace(PROJECT_ROOT=Path(d),
                installed_package_version=self.functions['installed_package_version'],
                dashboard_candidate=lambda: (_ for _ in ()).throw(RuntimeError('provenance reached')))
            with patch.dict(sys.modules, {'pricecatcher_config': cfg}), \
                 patch.object(self.prepare, 'saved_environment_records', return_value=[record]):
                with self.assertRaisesRegex(RuntimeError, 'provenance reached'):
                    self.prepare.verify_checkpoint()

    def test_exact_calendar_gaps_do_not_become_adjacent_weeks(self):
        make_features = definitions(ROOT / '05_features_and_labels.ipynb',
                                   ['make_features'], {'pd': pd})['make_features']
        weekly = pd.DataFrame({'week_start': pd.to_datetime(['2025-01-06', '2025-01-13', '2025-01-27']),
                               'weekly_median_price': [10., 11., 20.],
                               'record_count': [2, 2, 2], 'premise_count': [1, 1, 1]})
        result = make_features(weekly).set_index('week_start')
        self.assertTrue(pd.isna(result.loc['2025-01-13', 'next_week_price']))
        self.assertTrue(pd.isna(result.loc['2025-01-27', 'lag_1w_price']))
        self.assertTrue(pd.isna(result.loc['2025-01-27', 'next_week_price']))
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result.loc['2025-01-06', 'next_week_change_pct'], 10.)

    def test_ten_percent_boundary_and_missing_changes(self):
        result = self.functions['surge_mask'](np.array([10., 10. - 1e-12, 9.99, np.nan, -10.]), 10.)
        np.testing.assert_array_equal(result, [True, True, False, False, False])


if __name__ == '__main__':
    unittest.main()
