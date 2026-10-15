"""Synthetic pipeline checks; no network or downloaded research data required.

Run: python -m unittest discover -s tests -v
"""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="pricecatcher-test-")
        cls.environment = patch.dict(os.environ, {
            "PRICECATCHER_DATA_DIR": cls.temp.name,
            "PRICECATCHER_EXPERIMENT": "holdout",
        })
        cls.environment.start()
        import pricecatcher_config
        cls.cfg = importlib.reload(pricecatcher_config)
        cfg = cls.cfg
        cls.items = pd.DataFrame([
            {"item_code": code, "item": f"item {code}", "unit": "30 biji" if code in cfg.EGG_30_CODES
             else "10 biji" if code in cfg.ALL_EGG_CODES else "1kg", "item_group": group,
             "item_category": group} for code, group in cfg.PROJECT_ITEM_MAP.items()])
        cls.premises = pd.DataFrame([
            {"premise_code": shop, "premise": f"shop {shop}", "state": "Remote" if shop == 99 else "Sparse" if shop == 100 else "State A",
             "district": "Remote District" if shop == 99 else "Sparse District" if shop == 100 else "District A",
             "premise_type": "Borong" if shop == 99 else "Retail"}
            for shop in [1, 2, 3, 99, 100, 19117, 1291, 20872]])
        rows = []
        for date in pd.date_range("2025-01-01", "2026-06-30"):
            week = (date - pd.Timestamp("2025-01-01")).days // 7
            # Alternate one-/two-record weeks to exercise both coverage rejection reasons.
            if date.dayofweek == 0 or (date.dayofweek == 1 and week % 2 == 0):
                rows.append([date, 100, 1, 5.])
            for code in cfg.SCOPE_ITEM_CODES:
                # A later-appearing item must not become eligible in an earlier fold.
                if code == 992 and date < pd.Timestamp("2025-07-01"):
                    continue
                if code == 92 and pd.Timestamp("2025-04-07") <= date <= pd.Timestamp("2025-04-13"):
                    continue
                for shop in [1, 2, 3]:
                    if week % 2 == 0 and shop in [1, 2] and date.dayofweek > 0:
                        continue
                    if week % 2 == 1 and shop == 3 and date.dayofweek > 0:
                        continue
                    price = (5 + code % 5) * (1.25 if week % 5 == 0 else 1) * (4 if shop == 3 else 1)
                    if code in cfg.EGG_30_CODES:
                        price *= 3
                    rows.append([date, shop, code, price])
        for date, shop, code, price in [
            ("2025-02-27", 19117, 113, 55.), ("2025-02-28", 19117, 113, 55.),
            ("2025-03-01", 19117, 113, 55.), ("2025-03-02", 19117, 113, 55.),
            ("2025-09-29", 1291, 1589, 28.5), ("2025-09-29", 20872, 1589, 28.5),
            ("2025-01-03", 99, 1, 5.),
        ]:
            rows.append([pd.Timestamp(date), shop, code, price])
        cls.prices = pd.DataFrame(rows, columns=["date", "premise_code", "item_code", "price"])
        cls.real_read = pd.read_parquet

        def read_source(path, *args, **kwargs):
            if path == cfg.URL_ITEM:
                return cls.items.copy()
            if path == cfg.URL_PREMISE:
                return cls.premises.copy()
            if isinstance(path, str) and path.startswith("https://storage.data.gov.my/pricecatcher/pricecatcher_"):
                month = path.rsplit("_", 1)[1].removesuffix(".parquet")
                return cls.prices.loc[cls.prices.date.dt.to_period("M").eq(pd.Period(month))].copy()
            return cls.real_read(path, *args, **kwargs)

        cls.source_patch = patch.object(pd, "read_parquet", side_effect=read_source)
        cls.source_patch.start()
        cls.run_notebook("01_integrate_pricecatcher.ipynb")

    @classmethod
    def tearDownClass(cls):
        cls.source_patch.stop()
        cls.environment.stop()
        cls.temp.cleanup()

    @classmethod
    def run_notebook(cls, filename):
        notebook = nbformat.read(ROOT / filename, as_version=4)
        nbformat.validate(notebook)
        namespace = {"display": lambda *args: None, "__name__": "__main__"}
        with contextlib.redirect_stdout(io.StringIO()), patch.object(plt, "show", side_effect=lambda: plt.close("all")):
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    exec(compile(cell.source, f"{filename}:cell-{index}", "exec"), namespace)
        return namespace

    @classmethod
    def prepare(cls, experiment):
        os.environ["PRICECATCHER_EXPERIMENT"] = experiment
        for name in ["02_data_understanding_and_quality.ipynb", "03_cleaning_and_outliers.ipynb",
                     "04_weekly_aggregation.ipynb", "05_features_and_labels.ipynb"]:
            cls.run_notebook(name)
        return cls.cfg

    def test_all_experiments_and_coverage(self):
        for experiment in ["fold_1", "fold_2", "fold_3", "holdout"]:
            with self.subTest(experiment=experiment):
                cfg = self.prepare(experiment)
                settings = json.loads((cfg.STAGE5_DIR / "experiment_settings.json").read_text())
                self.assertEqual(settings["target_status"], settings["preparation"]["target_status"])
                self.assertEqual(settings["exported_splits"], ["train", "validation", "test"] if experiment == "holdout" else ["train", "validation"])
                for name in settings["exported_splits"]:
                    frame = pd.read_parquet(cfg.STAGE5_DIR / f"{name}.parquet")
                    cutoff = {"train": cfg.TRAIN_END_DATE, "validation": cfg.VALIDATION_END_DATE, "test": cfg.TEST_END_DATE}[name]
                    self.assertTrue(frame.next_week_end.le(pd.Timestamp(cutoff)).all())
                    if name != "train":
                        lower = cfg.TRAIN_END_DATE if name == "validation" else cfg.VALIDATION_END_DATE
                        self.assertTrue((frame.week_start + pd.Timedelta(days=6)).gt(pd.Timestamp(lower)).all())
                    self.assertFalse(any(c.startswith("next_") for c in settings["feature_columns"]))
                if experiment != "holdout":
                    self.assertFalse((cfg.STAGE5_DIR / "test.parquet").exists())
                daily = pd.read_csv(cfg.STAGE3_DIR / "daily_geographical_coverage.csv")
                self.assertTrue(daily.drop(columns=["state", "district", "source_records"]).sum(axis=1).eq(daily.source_records).all())
                self.assertEqual(daily.loc[daily.state.eq("Remote"), "retained_records"].iloc[0], 0)
                coverage = pd.read_csv(cfg.STAGE5_DIR / "district_coverage.csv")
                self.assertGreater(coverage.missing_next_week.sum(), 0)
                self.assertGreater(coverage.insufficient_history.sum(), 0)
                self.assertGreater(coverage.insufficient_current_coverage.sum(), 0)
                self.assertGreater(coverage.insufficient_next_coverage.sum(), 0)
                all_rows = coverage.loc[coverage.period.eq("all")]
                self.assertTrue(all_rows.assessable_rows.eq(all_rows.observed_current_weeks - all_rows.horizon_crosses_cutoff - all_rows.outcome_outside_scope).all())
                series = pd.read_csv(cfg.STAGE5_DIR / "series_coverage.csv")
                self.assertEqual(series.loc[series.state.eq("Remote"), "usable_weeks"].iloc[0], 0)
                self.assertGreater(series.missing_weeks_within_observed_span.sum(), 0)
                latest = pd.read_parquet(cfg.STAGE5_DIR / "latest_prediction_inputs.parquet")
                self.assertNotIn("surge_label", latest)
                self.assertGreater(len(latest), 0)
                reviewed = pd.read_csv(cfg.STAGE3_DIR / "documented_exclusions.csv")
                self.assertTrue(pd.to_datetime(reviewed.date).le(pd.Timestamp(cfg.TRAIN_END_DATE)).all())
                self.assertEqual(len(reviewed), 6 if experiment == "holdout" else 0)
                self.run_notebook("90_optional_source_diagnostics.ipynb")

    def test_future_prices_cannot_change_fold_training_preparation(self):
        cfg = self.prepare("fold_1")
        coverage = pd.read_csv(cfg.STAGE2_ITEM_COVERAGE_CSV)
        self.assertTrue(coverage.loc[coverage.item_code.eq(992), "exclude_low_coverage"].iloc[0])
        bounds = json.loads((cfg.STAGE3_DIR / "cleaning_summary.json").read_text())["training_price_bounds"]
        training = pd.read_parquet(cfg.STAGE5_DIR / "train.parquet")
        original = pd.read_parquet(cfg.STAGE1_COMBINED_PARQUET)
        changed = original.copy()
        changed.loc[changed.date.gt(pd.Timestamp(cfg.TRAIN_END_DATE)), "price"] *= 17
        changed.to_parquet(cfg.STAGE1_COMBINED_PARQUET, index=False)
        try:
            cfg = self.prepare("fold_1")
            pd.testing.assert_frame_equal(coverage, pd.read_csv(cfg.STAGE2_ITEM_COVERAGE_CSV))
            self.assertEqual(bounds, json.loads((cfg.STAGE3_DIR / "cleaning_summary.json").read_text())["training_price_bounds"])
            pd.testing.assert_frame_equal(training, pd.read_parquet(cfg.STAGE5_DIR / "train.parquet"))
        finally:
            original.to_parquet(cfg.STAGE1_COMBINED_PARQUET, index=False)

    def test_stale_stage_settings_are_rejected(self):
        cfg = self.prepare("fold_1")
        path = cfg.STAGE2_DIR / "preparation_settings.json"
        original = path.read_text()
        changed = json.loads(original)
        changed["low_coverage_ratio"] = .99
        path.write_text(json.dumps(changed))
        try:
            with self.assertRaisesRegex(ValueError, "settings/source changed"):
                self.run_notebook("03_cleaning_and_outliers.ipynb")
        finally:
            path.write_text(original)

    def test_calendar_gap_is_not_a_previous_week(self):
        from pricecatcher_features import make_features
        frame = pd.DataFrame({"week_start": pd.to_datetime(["2025-01-06", "2025-01-20", "2025-01-27"]),
                              "weekly_median_price": [10., 12., 15.], "record_count": [2, 2, 2], "premise_count": [1, 1, 1]})
        result = make_features(frame)
        self.assertTrue(pd.isna(result.loc[0, "next_week_price"]))
        self.assertTrue(pd.isna(result.loc[1, "lag_1w_price"]))
        self.assertEqual(result.loc[1, "next_week_change_pct"], 25.)
