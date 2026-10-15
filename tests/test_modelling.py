"""Model checks and an executed Notebook 06 using synthetic prepared folds."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import joblib
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd

import test_preparation as fixtures

ROOT = Path(__file__).resolve().parents[1]
# Load the notebook's function-only cell; no separate application helper module is needed.
_namespace = {}
for _cell in nbformat.read(ROOT / "06_model_training_and_evaluation.ipynb", as_version=4).cells:
    if _cell.cell_type == "code" and _cell.source.startswith("# Shared modelling calculations"):
        exec(compile(_cell.source, "notebook-modelling-functions", "exec"), _namespace)
for _name in ["build_pipeline", "choose_cutoff", "evaluate", "model_inputs", "grouped_metrics", "check_frame", "food_group_scores", "verify_saved_run"]:
    globals()[_name] = _namespace[_name]


class ModellingTests(unittest.TestCase):
    def test_training_only_imputation_and_unseen_categories(self):
        settings = {"feature_columns": ["price", "empty", "item_code"],
                    "numeric_features": ["price", "empty"], "categorical_features": ["item_code"]}
        train = pd.DataFrame({"price": [1., 2., 3., 100.], "empty": [np.nan] * 4, "item_code": [1] * 4})
        validation = pd.DataFrame({"price": [10000., np.nan], "empty": [10., np.nan], "item_code": [999, 999]})
        model = build_pipeline("Logistic Regression", {"C": 1.}, settings, [0, 0, 0, 1])
        model.fit(model_inputs(train, settings), [0, 0, 0, 1])
        imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
        self.assertEqual(imputer.statistics_[0], 2.5)
        self.assertTrue(np.isfinite(model.predict_proba(model_inputs(validation, settings))).all())
        self.assertEqual(imputer.statistics_[0], 2.5)
        booster = build_pipeline("XGBoost", {"max_depth": 3}, settings, [0, 0, 0, 1])
        self.assertEqual(booster.named_steps["model"].get_params()["scale_pos_weight"], 3.)

    def test_metric_definitions_and_unsupported_groups(self):
        result = evaluate([0, 1, 0, 1], [0., 0., 0., 0.], [0, 0, 0, 0])
        self.assertEqual(result["average_precision"], .5)
        self.assertEqual(result["f1"], 0.)
        self.assertEqual(result["fn"], 2)
        unsupported = evaluate([0, 0], [.2, .3], [0, 1])
        self.assertTrue(np.isnan(unsupported["average_precision"]))
        self.assertTrue(np.isnan(unsupported["recall"]))
        frame = pd.DataFrame({"project_group": ["Sugar", "Sugar", "Eggs", "Eggs"], "surge_label": [0, 0, 0, 1]})
        metrics = grouped_metrics(frame, [.1, .2, .3, .8], [0, 0, 0, 1], ["project_group"])
        self.assertEqual(metrics.loc[metrics.project_group.eq("Sugar"), "positives"].iloc[0], 0)

    def test_cutoff_uses_maximum_f1_with_conservative_tie_break(self):
        y = np.array([0, 1, 0, 0, 1, 1])
        scores = np.array([.1, .2, .3, .4, .5, .6])
        cutoff = choose_cutoff(y, scores)
        results = [(s, evaluate(y, scores, scores >= s)["f1"]) for s in np.unique(scores)]
        best_f1 = max(f1 for _, f1 in results)
        expected = max(s for s, f1 in results if np.isclose(f1, best_f1))
        self.assertEqual(cutoff, expected)
        with self.assertRaisesRegex(ValueError, "both validation classes"):
            choose_cutoff([0, 0], [.1, .2])

    def test_food_group_baseline_uses_training_only_and_has_fallback(self):
        training = pd.DataFrame({"project_group": ["Eggs", "Eggs", "Sugar", "Sugar"], "surge_label": [0, 1, 0, 0]})
        future = pd.DataFrame({"project_group": ["Eggs", "Sugar", "Unknown"], "surge_label": [1, 1, 1]})
        expected = np.array([.5, 0., .25])
        np.testing.assert_allclose(food_group_scores(training, future), expected)
        future["surge_label"] = 0
        np.testing.assert_allclose(food_group_scores(training, future), expected)

    def test_notebook_runs_and_saved_pipeline_reproduces_test_scores(self):
        fixtures.PreparationTests.setUpClass()
        try:
            for experiment in ["fold_1", "fold_2", "fold_3", "holdout"]:
                cfg = fixtures.PreparationTests.prepare(experiment)
            # The real notebook kernel must run the same Python interpreter/dependencies.
            kernel_root = cfg.PROJECT_ROOT / "test_jupyter"
            folder = kernel_root / "kernels" / "pricecatcher-test"
            folder.mkdir(parents=True)
            (folder / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "PriceCatcher test", "language": "python",
            }))
            notebook = nbformat.read(ROOT / "06_model_training_and_evaluation.ipynb", as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.source = cell.source.replace('RUN_MODE = "reuse"', 'RUN_MODE = "train"')
            with patch.dict(os.environ, {"JUPYTER_PATH": str(kernel_root) + os.pathsep + os.environ.get("JUPYTER_PATH", "")}):
                NotebookClient(notebook, timeout=240, kernel_name="pricecatcher-test",
                    resources={"metadata": {"path": str(ROOT)}}).execute()
            outputs = cfg.PROJECT_ROOT / "06_models"
            fold_metrics = pd.read_csv(outputs / "fold_metrics.csv")
            self.assertEqual(len(fold_metrics), 30)  # 8 candidates + 2 baselines, across 3 folds.
            self.assertEqual(fold_metrics.average_precision.notna().sum(), 30)
            selection = json.loads((outputs / "selection.json").read_text())
            settings = json.loads((cfg.STAGE5_DIR / "experiment_settings.json").read_text())
            test = pd.read_parquet(cfg.STAGE5_DIR / "test.parquet")
            model = joblib.load(outputs / "best_model.joblib")
            predictions = pd.read_parquet(outputs / "test_predictions.parquet")
            np.testing.assert_allclose(model.predict_proba(model_inputs(test, settings))[:, 1], predictions.model_score)
            np.testing.assert_array_equal(predictions.predicted_surge, (predictions.model_score >= selection["score_cutoff"]).astype(int))
            imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
            train = pd.read_parquet(cfg.STAGE5_DIR / "train.parquet")
            expected = train[settings["numeric_features"]].median().fillna(0).to_numpy()
            np.testing.assert_allclose(imputer.statistics_, expected)
            self.assertEqual(len(pd.read_csv(outputs / "benchmark_checks.csv")), 3)
            manifest = json.loads((outputs / "model_manifest.json").read_text())
            self.assertEqual(len(manifest["input_sha256"]), 14)
            self.assertEqual(len(manifest["code_sha256"]), 3)
            self.assertEqual(len(manifest["model_sha256"]), 64)
            self.assertTrue((outputs / "test_evaluation.png").exists())
            self.assertTrue((outputs / "supplementary" / "food_group_baseline.csv").exists())
            source_quality = pd.read_csv(outputs / "supplementary" / "validation_source_quality.csv")
            validation = pd.read_parquet(cfg.STAGE5_DIR / "validation.parquet")
            self.assertEqual(source_quality.rows.sum(), len(validation))
            import hashlib
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs.iterdir() if p.is_file()}
            reuse_notebook = nbformat.read(ROOT / "06_model_training_and_evaluation.ipynb", as_version=4)
            with patch.dict(os.environ, {"JUPYTER_PATH": str(kernel_root) + os.pathsep + os.environ.get("JUPYTER_PATH", "")}):
                NotebookClient(reuse_notebook, timeout=240, kernel_name="pricecatcher-test",
                    resources={"metadata": {"path": str(ROOT)}}).execute()
            after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs.iterdir() if p.is_file()}
            self.assertEqual(before, after)
            import subprocess
            runner = subprocess.run([sys.executable, str(ROOT / "prepare.py"), "--experiment", "all"],
                                    check=True, capture_output=True, text=True)
            self.assertEqual(runner.stdout.count("existing preparation reused"), 4)
            verify_saved_run(outputs, cfg.PROJECT_ROOT)
            # Deliberate refresh cannot overwrite inputs of an exposed completed experiment.
            blocked = subprocess.run([sys.executable, str(ROOT / "prepare.py"), "--experiment", "holdout", "--refresh"],
                                     capture_output=True, text=True)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("completed model exists", blocked.stderr)
            original_selection = (outputs / "selection.json").read_text()
            changed = json.loads(original_selection)
            changed["score_cutoff"] = .123456
            (outputs / "selection.json").write_text(json.dumps(changed))
            try:
                with self.assertRaisesRegex(ValueError, "Selection differs"):
                    verify_saved_run(outputs, cfg.PROJECT_ROOT)
            finally:
                (outputs / "selection.json").write_text(original_selection)
            self.assertNotIn("surge_label", pd.read_parquet(outputs / "latest_predictions.parquet"))
            broken = test.copy()
            broken.loc[broken.index[0], "next_week_end"] = pd.Timestamp("2027-01-01")
            with self.assertRaises(ValueError):
                check_frame(broken, settings, "test")
        finally:
            fixtures.PreparationTests.tearDownClass()
