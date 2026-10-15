"""Shared settings for Notebooks 01–05. Keep this file beside the notebooks."""
from pathlib import Path
from datetime import date
import os

# Outputs stay beside this file by default. Set PRICECATCHER_DATA_DIR to override.
PROJECT_ROOT = Path(os.environ.get("PRICECATCHER_DATA_DIR", Path(__file__).resolve().parent / "data")).expanduser().resolve()
# Change only this name for an interactive run, or use prepare.py.
EXPERIMENT = os.environ.get("PRICECATCHER_EXPERIMENT", "holdout")
START_MONTH = "2025-01"
END_MONTH = "2026-06"
# Each tuple is (training outcome cutoff, validation outcome cutoff, test cutoff).
# Folds expand within 2025; only holdout uses the final 2026 test period.
EXPERIMENTS = {
    "fold_1": ("2025-05-31", "2025-07-31", None),
    "fold_2": ("2025-07-31", "2025-09-30", None),
    "fold_3": ("2025-09-30", "2025-12-31", None),
    "holdout": ("2025-12-31", "2026-03-31", "2026-06-30"),
}
if EXPERIMENT not in EXPERIMENTS:
    raise ValueError(f"Unknown experiment {EXPERIMENT!r}; choose from {list(EXPERIMENTS)}")
TRAIN_END_DATE, VALIDATION_END_DATE, TEST_END_DATE = EXPERIMENTS[EXPERIMENT]
EVALUATION_END_DATE = TEST_END_DATE or VALIDATION_END_DATE
for cutoff in [TRAIN_END_DATE, VALIDATION_END_DATE, EVALUATION_END_DATE]:
    date.fromisoformat(cutoff)  # Reject invalid calendar dates immediately.
if not START_MONTH + "-01" < TRAIN_END_DATE < VALIDATION_END_DATE or EVALUATION_END_DATE[:7] > END_MONTH:
    raise ValueError("Experiment cutoffs must be ordered inside the source scope.")
if TEST_END_DATE is not None and not VALIDATION_END_DATE < TEST_END_DATE:
    raise ValueError("Test must follow validation.")

STAGE1_DIR = PROJECT_ROOT / "01_integrated"  # shared, unchanged source snapshot
RUN_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT
STAGE2_DIR = RUN_DIR / "02_quality"
STAGE3_DIR = RUN_DIR / "03_cleaned"
STAGE4_DIR = RUN_DIR / "04_weekly"
STAGE5_DIR = RUN_DIR / "05_modelling"
LOOKUP_DIR = STAGE1_DIR / "lookups"
for folder in [STAGE1_DIR, STAGE2_DIR, STAGE3_DIR, STAGE4_DIR, STAGE5_DIR, LOOKUP_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

BASE_PRICE_URL = "https://storage.data.gov.my/pricecatcher/pricecatcher_{month}.parquet"
URL_ITEM = "https://storage.data.gov.my/pricecatcher/lookup_item.parquet"
URL_PREMISE = "https://storage.data.gov.my/pricecatcher/lookup_premise.parquet"

PROJECT_ITEM_MAP = {
    1: "Chicken", 2: "Chicken",
    118: "Eggs", 119: "Eggs", 120: "Eggs",
    1109: "Eggs", 1110: "Eggs", 1111: "Eggs",
    904: "Rice", 992: "Rice", 1581: "Rice", 1582: "Rice",
    918: "Cooking Oil", 1091: "Cooking Oil", 1094: "Cooking Oil", 1100: "Cooking Oil",
    92: "Vegetables", 93: "Vegetables", 94: "Vegetables",
    113: "Vegetables", 114: "Vegetables", 1556: "Vegetables", 1558: "Vegetables",
    917: "Flour", 1593: "Flour", 1589: "Sugar", 1590: "Sugar",
}
SCOPE_ITEM_CODES = list(PROJECT_ITEM_MAP)
EGG_30_CODES = [1109, 1110, 1111]
ALL_EGG_CODES = [118, 119, 120, 1109, 1110, 1111]
TEXT_COLS = ["state", "district", "premise_type", "premise", "project_group",
             "item", "unit", "item_group", "item_category"]
FINAL_COLS = ["date", "state", "district", "premise_type", "premise", "premise_code",
              "project_group", "item_code", "item", "unit", "item_group", "item_category", "price"]
GROUP_KEYS = ["project_group", "item_code", "item", "unit", "state", "district", "premise_type"]

# Training-only item screening and price flags.
LOW_COVERAGE_RATIO = 0.05
REMOVE_BORONG = True
IQR_MULTIPLIER = 6
LOW_MEDIAN_MULTIPLIER = 0.20
HIGH_MEDIAN_MULTIPLIER = 5.00
# Review dates were not recorded: avoid applying retrospective exceptions to early folds.
APPLY_DOCUMENTED_EXCLUSIONS = EXPERIMENT == "holdout"
REMOVE_FLAGGED_PRICES = False  # A statistical flag does not confirm an error.

# Common next-week surge definition: project setting, not an official standard.
SURGE_THRESHOLD_PCT = 10.0
TARGET_STATUS = "working research definition: observed local median, not a fixed-shop price index"
MIN_RECORDS = 2
MIN_PREMISES = 1
MIN_HISTORY_4W = 2
# Comparison only: these do not change the main target or exported splits.
COMPARISON_MIN_PREMISES = 2
COMPARISON_MIN_DAYS = 3

# Only necessary files are named here.
STAGE1_COMBINED_PARQUET = STAGE1_DIR / "integrated_prices.parquet"
STAGE2_ITEM_COVERAGE_CSV = STAGE2_DIR / "training_item_coverage.csv"
STAGE3_MODEL_READY_PARQUET = STAGE3_DIR / "daily_prices.parquet"
STAGE4_WEEKLY_PARQUET = STAGE4_DIR / "weekly_prices.parquet"

# Saved at every stage; changing settings requires rerunning 02–05 for this experiment.
PREPARATION_SETTINGS = {
    "schema_version": 2, "experiment": EXPERIMENT,
    "source_start_month": START_MONTH, "source_end_month": END_MONTH,
    "train_end": TRAIN_END_DATE, "validation_end": VALIDATION_END_DATE,
    "test_end": TEST_END_DATE, "evaluation_end": EVALUATION_END_DATE,
    "item_map": {str(code): group for code, group in PROJECT_ITEM_MAP.items()},
    "low_coverage_ratio": LOW_COVERAGE_RATIO, "remove_borong": REMOVE_BORONG,
    "iqr_multiplier": IQR_MULTIPLIER, "low_median_multiplier": LOW_MEDIAN_MULTIPLIER,
    "high_median_multiplier": HIGH_MEDIAN_MULTIPLIER,
    "remove_flagged_prices": REMOVE_FLAGGED_PRICES,
    "apply_documented_exclusions": APPLY_DOCUMENTED_EXCLUSIONS,
    "documented_exclusion_policy": "fixed retrospective exceptions, holdout training only by default; v1-six-keys",
    "minimum_records": MIN_RECORDS, "minimum_premises": MIN_PREMISES,
    "minimum_history_4w": MIN_HISTORY_4W,
    "comparison_minimum_premises": COMPARISON_MIN_PREMISES,
    "comparison_minimum_days": COMPARISON_MIN_DAYS,
    "surge_threshold_pct": SURGE_THRESHOLD_PCT, "target_status": TARGET_STATUS,
    "aggregation": "median of daily observations; equal-premise comparator only",
    "imputation": "none in preparation; fit feature imputation within each model training fold",
}
