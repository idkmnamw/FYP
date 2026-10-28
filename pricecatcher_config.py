"""Shared settings and run locations. Keep this file beside the project notebooks."""
from pathlib import Path
from datetime import date, datetime
import hashlib
import json
import os
import shutil
from zoneinfo import ZoneInfo
import numpy as np

# Notebooks share one active run across separate kernels. Advanced/runner paths
# still use PRICECATCHER_DATA_DIR explicitly and never depend on this pointer.
DATA_HOME = Path(__file__).resolve().parent / "data"
ACTIVE_RUN_FILE = DATA_HOME / "active_run.txt"


def find_run_folder(base):
    """Find the latest local run by folder date, never by model score."""
    for run in sorted((base / "runs").glob("run_*"), reverse=True):
        if not run.is_dir():
            continue
        if any((run / model / filename).is_file()
               for model in ["06_models", "06_models_reproduction"]
               for filename in ["best_model.joblib", "model_manifest.json"]):
            return run
        source = run / "01_integrated"
        if (source / "integrated_prices.parquet").is_file() and (source / "source_manifest.json").is_file():
            return run
    return base


def current_run_folder():
    """An explicit path wins; otherwise continue the run started by Notebook 01."""
    if os.environ.get("PRICECATCHER_DATA_DIR"):
        return Path(os.environ["PRICECATCHER_DATA_DIR"]).expanduser().resolve()
    if ACTIVE_RUN_FILE.exists():
        relative = Path(ACTIVE_RUN_FILE.read_text().strip())
        if relative.is_absolute() or relative.parts[:1] != ("runs",) or ".." in relative.parts:
            raise ValueError("Invalid data/active_run.txt. It must contain a relative runs/run_... path.")
        return DATA_HOME / relative
    return find_run_folder(DATA_HOME)


PROJECT_ROOT = current_run_folder()
# The terminal runner sets this internally; interactive notebooks use holdout.
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
# Percentage-point tolerance for floating-point arithmetic, not an economic margin.
LABEL_TOLERANCE_PCT = 1e-10
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
MODEL_DIRS = (PROJECT_ROOT / "06_models", PROJECT_ROOT / "06_models_reproduction")


def surge_mask(change_pct, threshold_pct, tolerance_pct=LABEL_TOLERANCE_PCT):
    """Count an exact threshold rise despite tiny floating-point rounding errors."""
    return (change_pct >= threshold_pct) | np.isclose(
        change_pct, threshold_pct, rtol=0, atol=tolerance_pct)


def protect_frozen_preparation():
    """Keep direct notebook runs from replacing inputs of an existing model."""
    if (any((folder / "best_model.joblib").exists() for folder in MODEL_DIRS)
            or any(PROJECT_ROOT.glob("0[6-9]_*/*/study_manifest.json"))):
        raise ValueError("A frozen model exists in this run. To reproduce 01–10, Run All in Notebook 01 to start a separate run, then continue the numbered workflow. To view results, open the matching saved model notebook. Use --rerun-all for a separate reproduction.")


def verify_source_snapshot(source_dir=None):
    """Check the actual integrated data/lookups before preparing a new experiment."""
    source_dir = Path(source_dir) if source_dir is not None else STAGE1_DIR
    manifest_path = source_dir / "source_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source snapshot missing: {manifest_path}. Run Notebook 01 first.")
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("source_files_sha256", {})
    required = {"integrated_prices.parquet", "lookups/lookup_item.parquet", "lookups/lookup_premise.parquet"}
    if set(files) != required:
        raise ValueError("Source file hashes are missing/incomplete. Restore the matching source snapshot, or run 01 for a new study.")
    for relative_path, expected_hash in files.items():
        path = source_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Source file missing: {relative_path}. Restore the matching snapshot.")
        with path.open("rb") as handle:
            actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Source file changed: {relative_path}. Restore the matching snapshot, or run 01 in a separate folder for a new study.")
    snapshot = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if snapshot != manifest.get("source_snapshot_sha256"):
        raise ValueError("Source snapshot ID disagrees with its file hashes. Restore the original manifest.")
    return manifest

def create_run_folder(base):
    """Create a separate reproduction and copy a verified local source if available."""
    base = Path(base)
    candidates = [PROJECT_ROOT / "01_integrated", base / "01_integrated"]
    candidates += sorted((base / "runs").glob("run_*/01_integrated"), reverse=True)
    source = next((path for path in candidates if (path / "integrated_prices.parquet").is_file()
                   and (path / "source_manifest.json").is_file()), None)
    if source is not None:
        verify_source_snapshot(source)
    stamp = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).strftime("run_%Y%m%d_%H%M%S_%f")
    run = base / "runs" / stamp
    run.mkdir(parents=True, exist_ok=False)
    if source is not None:
        shutil.copytree(source, run / "01_integrated", ignore=shutil.ignore_patterns("executed"))
    return run


def remember_run(run):
    """Save a relative folder pointer; never change a model or its prepared inputs."""
    relative = Path(run).relative_to(DATA_HOME)
    DATA_HOME.mkdir(parents=True, exist_ok=True)
    temporary = ACTIVE_RUN_FILE.with_suffix(".tmp")
    temporary.write_text(str(relative))
    temporary.replace(ACTIVE_RUN_FILE)


def start_notebook_run():
    """Run All in 01 starts a new run; explicit runner paths stay unchanged."""
    if os.environ.get("PRICECATCHER_DATA_DIR"):
        return PROJECT_ROOT
    run = create_run_folder(DATA_HOME)
    remember_run(run)
    print("Started a new reproduction. Previous experiments are preserved.")
    return run


# Saved at every stage; changing settings requires rerunning 02–05 for this experiment.
PREPARATION_SETTINGS = {
    "schema_version": 3, "experiment": EXPERIMENT,
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
    "label_comparison": "greater_equal_with_absolute_tolerance",
    "label_tolerance_pct": LABEL_TOLERANCE_PCT,
    "aggregation": "median of daily observations; equal-premise comparator only",
    "imputation": "none in preparation; fit feature imputation within each model training fold",
}


def notebook_path(filename):
    """Locate a main or research notebook without depending on the kernel folder."""
    repository = Path(__file__).resolve().parent
    for path in [repository / filename, repository / "research" / filename]:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Notebook missing from the FYP project: {filename}")


def latest_longer_history(model_dir):
    """Find the latest completed compatible 12 run by completion time, never score."""
    model_dir = Path(model_dir)
    original = json.loads((model_dir / "model_manifest.json").read_text())
    selection = json.loads((model_dir / "selection.json").read_text())
    model_key = str((model_dir / "best_model.joblib").resolve())
    candidates = []
    for path in (PROJECT_ROOT / "12_longer_history_experiment").glob("*/experiment_manifest.json"):
        record = json.loads(path.read_text())
        protocol = record["protocol"]
        if (protocol["candidate"] == selection["candidate"]
                and protocol["cutoff"] == selection["score_cutoff"]
                and protocol["input_sha256"].get(model_key) == original["model_sha256"]
                and record.get("prepared_test_accessed") is False):
            candidates.append(path)
    return max(candidates, key=lambda path: path.stat().st_mtime_ns).parent if candidates else None


def input_reference(path):
    """Record new inputs relative to the run or repository, not a user's home folder."""
    path = Path(path).resolve()
    for prefix, root in [("run::", PROJECT_ROOT), ("repo::", Path(__file__).parent)]:
        if path.is_relative_to(root.resolve()):
            return prefix + path.relative_to(root.resolve()).as_posix()
    raise ValueError(f"Keep study inputs inside the run or repository: {path}")


def resolve_input(reference):
    """Locate moved inputs; callers still verify their original SHA-256 hashes."""
    reference = str(reference)
    for prefix, root in [("run::", PROJECT_ROOT), ("repo::", Path(__file__).parent)]:
        if reference.startswith(prefix):
            path = (root / reference.removeprefix(prefix)).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Input reference escapes its saved root.")
            return path
    path = Path(reference)
    if not path.is_absolute():
        return resolve_input("run::" + reference)
    # Legacy manifests are immutable. Relocate only their explicit data/run suffixes.
    parts = path.parts
    for i, part in enumerate(parts[:-1]):
        if part == "runs" and parts[i + 1].startswith("run_"):
            return (PROJECT_ROOT / Path(*parts[i + 2:])).resolve()
    if "data" in parts:
        i = len(parts) - 1 - parts[::-1].index("data")
        suffix = Path(*parts[i + 1:])
        active = (suffix.parts and (suffix.parts[0] in {"experiments", "01_integrated", "11_deployment", "11_anomalies"}
                  or suffix.parts[0].startswith(("06_", "07_", "08_", "09_", "10_", "12_longer_history_experiment", "13_registered_"))))
        return ((PROJECT_ROOT if active else DATA_HOME) / suffix).resolve()
    if path.is_file():
        return path  # An unchanged external legacy input; no guessed replacement.
    raise FileNotFoundError(f"Missing saved input: {reference}. Copy its matching source/cache too.")


def verify_saved_study(folder):
    """Verify a locally trained study before inference; never fit or repair it."""
    folder = Path(folder).resolve()
    if not folder.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Study must belong to the active run.")
    record = json.loads((folder / "study_manifest.json").read_text())
    if record.get("status") != "completed":
        raise ValueError("The selected study is incomplete. Restore its completed outputs.")
    def check(path, expected):
        with Path(path).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen study input/output changed: {path}")
    for name, expected in record["output_sha256"].items():
        check(folder / name, expected)
    for name, expected in record["input_sha256"].items():
        check(PROJECT_ROOT / name, expected)
    for name, expected in record.get("additional_input_sha256", {}).items():
        check(resolve_input(name), expected)
    selection = json.loads((folder / "selection.json").read_text())
    if selection != record["selection"]:
        raise ValueError("Study selection and manifest disagree.")
    cutoff = selection.get("score_cutoff")
    if not isinstance(cutoff, (int, float)) or not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Selected cutoff must be a finite model score in [0, 1].")
    if selection["surge_threshold_pct"] != 10:
        raise ValueError("The dashboard requires the existing 10% surge definition.")
    return record


def dashboard_candidate():
    """Pin 10's declared candidate once; later comparisons cannot silently replace it."""
    import csv
    pin_path = PROJECT_ROOT / "11_deployment" / "deployment_selection.json"
    def digest(path):
        with Path(path).open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    if pin_path.exists():
        pin = json.loads(pin_path.read_text())
    else:
        comparisons = sorted((PROJECT_ROOT / "10_model_comparison").glob("*/comparison_manifest.json"), reverse=True)
        if not comparisons:
            raise FileNotFoundError("Complete Notebook 10 before deployment stages 11–13.")
        comparison_path = comparisons[0]
        comparison = json.loads(comparison_path.read_text())
        if comparison.get("status") != "completed" or comparison.get("test_evaluated") is not False:
            raise ValueError("A completed exploratory comparison is required.")
        for name, expected in comparison["output_sha256"].items():
            if digest(comparison_path.parent / name) != expected:
                raise ValueError("Saved comparison outputs changed.")
        with (comparison_path.parent / "model_comparison.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        winner = sorted(rows, key=lambda r: (-float(r["mean_fold_ap"]), float(r["std_fold_ap"]), r["model"]))[0]["model"]
        if winner != comparison["candidate_under_fold_rule"]:
            raise ValueError("Comparison selection differs from the declared fold rule.")
        source = comparison["studies"][winner]
        study_path = PROJECT_ROOT / source["folder"] / "study_manifest.json"
        if digest(study_path) != source["manifest_sha256"]:
            raise ValueError("Selected study changed after model comparison.")
        record = verify_saved_study(study_path.parent)
        selection = record["selection"]
        identity = {"study_sha256": digest(study_path), "cutoff": selection["score_cutoff"]}
        pin = {"schema_version": 1, "candidate": winner,
               "deployment_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16],
               "study_manifest": str(study_path.relative_to(PROJECT_ROOT)),
               "study_manifest_sha256": digest(study_path),
               "comparison_manifest": str(comparison_path.relative_to(PROJECT_ROOT)),
               "comparison_manifest_sha256": digest(comparison_path),
               "score_cutoff": selection["score_cutoff"], "status": "frozen exploratory deployment candidate"}
        pin_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = pin_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(pin, indent=2))
        temporary.replace(pin_path)
    study_path = PROJECT_ROOT / pin["study_manifest"]
    if digest(study_path) != pin["study_manifest_sha256"] or digest(PROJECT_ROOT / pin["comparison_manifest"]) != pin["comparison_manifest_sha256"]:
        raise ValueError("Pinned deployment provenance changed. Restore the matching snapshot; do not change hashes.")
    frozen_comparison = json.loads((PROJECT_ROOT / pin["comparison_manifest"]).read_text())
    comparison_folder = (PROJECT_ROOT / pin["comparison_manifest"]).parent
    for name, expected in frozen_comparison["output_sha256"].items():
        if digest(comparison_folder / name) != expected:
            raise ValueError("Frozen comparison outputs changed. Restore their matching originals.")
    if pin["candidate"] != frozen_comparison["candidate_under_fold_rule"]:
        raise ValueError("Pinned candidate differs from its frozen comparison decision.")
    declared = frozen_comparison["studies"][pin["candidate"]]
    expected_path = (PROJECT_ROOT / declared["folder"] / "study_manifest.json").resolve()
    identity = {"study_sha256": pin["study_manifest_sha256"], "cutoff": pin["score_cutoff"]}
    expected_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    if (expected_path != study_path.resolve() or declared["manifest_sha256"] != pin["study_manifest_sha256"]
            or pin["deployment_id"] != expected_id):
        raise ValueError("Pinned deployment identity or source study changed.")
    record = verify_saved_study(study_path.parent)
    if pin["score_cutoff"] != record["selection"]["score_cutoff"]:
        raise ValueError("Pinned score cutoff changed.")
    return pin, study_path.parent, record


def dashboard_directory():
    """Follow a completed deployment publication, or retain the legacy dashboard."""
    base = PROJECT_ROOT / "08_dashboard"
    pointer = base / "active_deployment.json"
    if not pointer.exists():
        return base
    record = json.loads(pointer.read_text())
    folder = (base / record["directory"]).resolve()
    if not folder.is_relative_to((base / "deployments").resolve()):
        raise ValueError("Invalid dashboard deployment directory.")
    return folder
