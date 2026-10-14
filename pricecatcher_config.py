"""Shared settings for Notebooks 01–05. Keep this file beside the notebooks."""
from pathlib import Path
import os

# Outputs stay beside this file by default. Set PRICECATCHER_DATA_DIR to override.
PROJECT_ROOT = Path(os.environ.get("PRICECATCHER_DATA_DIR", Path(__file__).resolve().parent / "data")).expanduser().resolve()
STAGE1_DIR = PROJECT_ROOT / "01_integrated"
STAGE2_DIR = PROJECT_ROOT / "02_quality"
STAGE3_DIR = PROJECT_ROOT / "03_cleaned"
STAGE4_DIR = PROJECT_ROOT / "04_weekly"
STAGE5_DIR = PROJECT_ROOT / "05_modelling"
LOOKUP_DIR = STAGE1_DIR / "lookups"
for folder in [STAGE1_DIR, STAGE2_DIR, STAGE3_DIR, STAGE4_DIR, STAGE5_DIR, LOOKUP_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

START_MONTH = "2025-01"
END_MONTH = "2026-06"
TRAIN_END_DATE = "2025-12-31"
VALIDATION_END_DATE = "2026-03-31"

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
