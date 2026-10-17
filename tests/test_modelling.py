"""Model checks and an executed Notebook 06 using synthetic prepared folds."""
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import joblib
import nbformat
import numpy as np
import pandas as pd

import test_preparation as fixtures

ROOT = Path(__file__).resolve().parents[1]
# Load the notebook's function-only cell; no separate application helper module is needed.
_namespace = {}
for _cell in nbformat.read(ROOT / "06_model_training_and_evaluation.ipynb", as_version=4).cells:
    if _cell.cell_type == "code" and _cell.source.startswith("# Shared modelling calculations"):
        exec(compile(_cell.source, "notebook-modelling-functions", "exec"), _namespace)
for _name in ["build_pipeline", "choose_cutoff", "evaluate", "model_inputs", "grouped_metrics", "check_frame", "food_group_scores", "verify_saved_run", "source_snapshot_status", "save_frozen_model"]:
    globals()[_name] = _namespace[_name]


class ModellingTests(unittest.TestCase):
    def test_checkpoint_failure_does_not_activate_an_unrecoverable_model(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            # Fail after writing selection, while writing the manifest.
            original_write = _namespace["write_json"]

            def fail_manifest(path, value):
                if path.name == "model_manifest.json":
                    raise OSError("simulated interrupted checkpoint")
                original_write(path, value)

            with patch.dict(_namespace, {"write_json": fail_manifest}):
                with self.assertRaisesRegex(OSError, "interrupted checkpoint"):
                    save_frozen_model({"fitted": True}, {"score_cutoff": .6}, {}, output)
            self.assertFalse((output / "best_model.joblib").exists())
            # A retry before test access is allowed because no model was activated.
            save_frozen_model({"fitted": True}, {"score_cutoff": .6}, {"input_sha256": {}}, output)
            manifest, selection = verify_saved_run(output, output)
            self.assertEqual(manifest["run_status"], "evaluation_pending")
            self.assertEqual(selection["score_cutoff"], .6)
            with self.assertRaisesRegex(ValueError, "frozen model already exists"):
                save_frozen_model({}, selection, {}, output)

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

    def test_snapshot_identity_ignores_manifest_timestamps_but_not_content(self):
        experiments = {
            "fold_1": {"preparation": {"source_manifest_sha256": "a" * 64, "source_snapshot_sha256": "same-content"}},
            "holdout": {"preparation": {"source_manifest_sha256": "b" * 64, "source_snapshot_sha256": "same-content"}},
        }
        self.assertEqual(source_snapshot_status(experiments, "train")["status"], "matching_content_snapshot")
        experiments["fold_1"]["preparation"]["source_snapshot_sha256"] = "different-content"
        with self.assertRaisesRegex(ValueError, "not consistently identified"):
            source_snapshot_status(experiments, "train")
        self.assertEqual(source_snapshot_status(experiments, "reuse")["status"], "historical_source_consistency_unresolved")

    def test_legacy_snapshot_mismatch_is_not_silently_declared_equivalent(self):
        experiments = {
            "fold_1": {"preparation": {"source_manifest_sha256": "a" * 64}},
            "holdout": {"preparation": {"source_manifest_sha256": "b" * 64}},
        }
        with self.assertRaises(ValueError):
            source_snapshot_status(experiments, "train")
        self.assertEqual(source_snapshot_status(experiments, "reuse")["status"], "historical_source_consistency_unresolved")

    def test_notebook_runs_and_saved_pipeline_reproduces_test_scores(self):
        fixtures.PreparationTests.setUpClass()
        try:
            cfg = fixtures.PreparationTests.cfg
            # Real Jupyter execution of 02–06 and optional 90 across all synthetic folds.
            # 01 was exercised above against synthetic monthly sources; no network is needed.
            completed = subprocess.run(
                [sys.executable, str(ROOT / "prepare.py"), "--experiment", "all", "--model-mode", "train", "--diagnostics"],
                check=True, capture_output=True, text=True)
            self.assertEqual(completed.stdout.count("90_optional_source_diagnostics.ipynb"), 4)
            for name in ["fold_1", "fold_2", "fold_3", "holdout"]:
                executed = cfg.PROJECT_ROOT / "experiments" / name / "executed"
                for filename in ["02_data_understanding_and_quality.ipynb", "03_cleaning_and_outliers.ipynb",
                                 "04_weekly_aggregation.ipynb", "05_features_and_labels.ipynb", "90_optional_source_diagnostics.ipynb"]:
                    notebook = nbformat.read(executed / filename, as_version=4)
                    self.assertTrue(all(c.execution_count is not None for c in notebook.cells if c.cell_type == "code"))
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
            self.assertEqual(manifest["run_status"], "completed")
            self.assertTrue((outputs / "test_evaluation.png").exists())
            self.assertTrue((outputs / "supplementary" / "food_group_baseline.csv").exists())
            source_quality = pd.read_csv(outputs / "supplementary" / "validation_source_quality.csv")
            validation = pd.read_parquet(cfg.STAGE5_DIR / "validation.parquet")
            self.assertEqual(source_quality.rows.sum(), len(validation))
            import hashlib
            # Emulate loss of evaluation exports after a frozen checkpoint.
            # Only this synthetic run's status is changed; real runs are untouched.
            frozen_model = (outputs / "best_model.joblib").read_bytes()
            frozen_selection = (outputs / "selection.json").read_bytes()
            manifest["run_status"] = "evaluation_pending"
            (outputs / "model_manifest.json").write_text(json.dumps(manifest))
            (outputs / "test_predictions.parquet").unlink()
            (outputs / "test_metrics.csv").unlink()
            subprocess.run([sys.executable, str(ROOT / "prepare.py"), "--model-mode", "reuse"],
                           check=True, capture_output=True, text=True)
            self.assertEqual((outputs / "best_model.joblib").read_bytes(), frozen_model)
            self.assertEqual((outputs / "selection.json").read_bytes(), frozen_selection)
            self.assertEqual(json.loads((outputs / "model_manifest.json").read_text())["run_status"], "completed")
            pd.testing.assert_frame_equal(predictions, pd.read_parquet(outputs / "test_predictions.parquet"))
            self.assertTrue((outputs / "test_metrics.csv").exists())
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs.iterdir() if p.is_file()}
            subprocess.run([sys.executable, str(ROOT / "prepare.py"), "--experiment", "all", "--model-mode", "reuse"],
                           check=True, capture_output=True, text=True)
            after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs.iterdir() if p.is_file()}
            self.assertEqual(before, after)
            # Restoring legacy model settings can leave newer weekly provenance.
            # Accept provenance differences only with matching policies/prices/labels.
            weekly_settings_path = cfg.STAGE4_DIR / "preparation_settings.json"
            original_weekly_settings = weekly_settings_path.read_bytes()
            audit_settings = json.loads(original_weekly_settings)
            audit_settings["source_manifest_sha256"] = "newer-audit-manifest"
            audit_settings["source_snapshot_sha256"] = "newer-audit-snapshot"
            weekly_settings_path.write_text(json.dumps(audit_settings))
            try:
                subprocess.run([sys.executable, str(ROOT / "prepare.py"), "--model-mode", "reuse"],
                               check=True, capture_output=True, text=True)
                audit = json.loads((outputs / "supplementary" / "analysis_settings.json").read_text())
                self.assertFalse(audit["weekly_audit_provenance"]["metadata_match"])
                self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs.iterdir() if p.is_file()})
                audit_settings["minimum_records"] += 1
                weekly_settings_path.write_text(json.dumps(audit_settings))
                rejected = subprocess.run([sys.executable, str(ROOT / "prepare.py"), "--model-mode", "reuse"],
                                          capture_output=True, text=True)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("Weekly audit policies differ", rejected.stderr)
            finally:
                weekly_settings_path.write_bytes(original_weekly_settings)
            weekly_path = cfg.STAGE4_WEEKLY_PARQUET
            original_weekly = weekly_path.read_bytes()
            changed_weekly = pd.read_parquet(weekly_path)
            row = validation.iloc[0]
            pair = changed_weekly.week_start.eq(row.week_start)
            for key in cfg.GROUP_KEYS:
                pair &= changed_weekly[key].eq(row[key])
            self.assertEqual(int(pair.sum()), 1)
            changed_weekly.loc[pair, "weekly_median_price"] *= 1.1
            changed_weekly.to_parquet(weekly_path, index=False)
            try:
                rejected = subprocess.run([sys.executable, str(ROOT / "prepare.py"), "--model-mode", "reuse"],
                                          capture_output=True, text=True)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("Weekly audit prices differ", rejected.stderr)
            finally:
                weekly_path.write_bytes(original_weekly)
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
            missing = cfg.STAGE5_DIR / "train.parquet"
            temporary = missing.with_suffix(".temporarily-missing")
            missing.rename(temporary)
            try:
                with self.assertRaisesRegex(FileNotFoundError, "Restore the original prepared folder"):
                    verify_saved_run(outputs, cfg.PROJECT_ROOT)
            finally:
                temporary.rename(missing)
            self.assertNotIn("surge_label", pd.read_parquet(outputs / "latest_predictions.parquet"))
            # Emulate a completed legacy study whose differing integration manifests were frozen at training time.
            # This is a synthetic fixture only: no real recorded hashes are rewritten.
            legacy_manifest = json.loads((outputs / "model_manifest.json").read_text())
            legacy_manifest.pop("source_provenance", None)
            for index, name in enumerate(["fold_1", "fold_2", "fold_3", "holdout"]):
                experiment_root = cfg.PROJECT_ROOT / "experiments" / name
                settings_path = experiment_root / "05_modelling" / "experiment_settings.json"
                legacy_settings = json.loads(settings_path.read_text())
                legacy_settings["preparation"].pop("source_snapshot_sha256", None)
                legacy_settings["preparation"]["source_manifest_sha256"] = str(index) * 64
                settings_path.write_text(json.dumps(legacy_settings))
                weekly_path = experiment_root / "04_weekly" / "preparation_settings.json"
                weekly_path.write_text(json.dumps(legacy_settings["preparation"]))
                legacy_manifest["input_sha256"][f"{name}/experiment_settings.json"] = hashlib.sha256(settings_path.read_bytes()).hexdigest()
                if name == "holdout":
                    legacy_manifest["holdout_preparation"] = legacy_settings["preparation"]
            (outputs / "model_manifest.json").write_text(json.dumps(legacy_manifest))
            frozen_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs.iterdir() if p.is_file()}
            subprocess.run([sys.executable, str(ROOT / "prepare.py"), "--model-mode", "reuse"],
                           check=True, capture_output=True, text=True)
            self.assertEqual(frozen_files, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs.iterdir() if p.is_file()})
            status = json.loads((outputs / "supplementary" / "source_provenance.json").read_text())
            self.assertEqual(status["status"], "historical_source_consistency_unresolved")
            broken = test.copy()
            broken.loc[broken.index[0], "next_week_end"] = pd.Timestamp("2027-01-01")
            with self.assertRaises(ValueError):
                check_frame(broken, settings, "test")
        finally:
            fixtures.PreparationTests.tearDownClass()
