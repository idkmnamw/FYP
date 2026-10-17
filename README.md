# PriceCatcher FYP

Only the main workflow is kept here: notebooks **01–06**, `pricecatcher_config.py`, `prepare.py`, `requirements.txt`, this README and `.gitignore`. Generated data/models stay out of Git. No extra application modules are required.

## Install once

Use Python 3.11 or newer, from the FYP folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For Jupyter, run `python -m jupyter lab` and select this environment as the kernel. Restore an older saved model using the package versions in its own `model_manifest.json`.

## Rerun the complete workflow

```bash
python prepare.py --rerun-all
```

Repeat this command whenever a complete rerun is needed. It:

1. Creates a separate folder under `data/runs/run_<timestamp>/`, preserving earlier results.
2. Copies an existing source snapshot, if available, and verifies its hashes. It can use a source cached by an earlier complete run.
3. Executes Notebook 01 using that cached source, then 02–05 separately for each expanding fold and holdout, then 06.
4. Saves executed notebooks, settings, coverage audits, models and metrics inside the new run folder.

With a complete cached source, no monthly data download is needed. Without one, the first run needs access to `storage.data.gov.my`. That download is currently blocked by this cloud's proxy (403); it can work from a Mac with normal access. Missing dependencies, inaccessible sources or changed/corrupt files produce an explicit error rather than silently accepting incomplete results.

The configured April–June 2026 test has already been inspected. Every rebuild is a **reproduction**, not fresh independent validation. Do not tune against reproduced test results. Keep the source snapshot, environment and run folder for results cited in the report.

## Reopen a completed run without retraining

Use the exact run folder printed by the command:

```bash
PRICECATCHER_DATA_DIR="/path/to/FYP/data/runs/run_<timestamp>" python prepare.py --model-mode auto
```

For original files stored directly under `data/`, use `python prepare.py --model-mode auto`. Notebook 06's default `RUN_MODE = "auto"` also reuses an existing local model, or trains into a separate reproduction folder if no model/manifest exists. Missing or outdated fold preparation can be generated from local integrated data when no frozen model protects it.

A partial saved run is never silently replaced. Reuse verifies the model, selection and prepared-input hashes. It may recover holdout metadata only when the entire reconstructed settings file has the exact original recorded hash and every other frozen input passes; the changed JSON is backed up. Model/Parquet files and expected hashes are preserved. A pending evaluation can resume with the frozen model/cutoff. Direct runs of 01–05 stop when they would replace a frozen run's inputs; use `--rerun-all` instead.

## What each notebook does

| Notebook | Purpose |
|---|---|
| 01 | Integrate sources, validate joins/units and preserve the source snapshot |
| 02 | Screen item coverage using training data only |
| 03 | Clean invalid records, flag suspicious prices and audit geographical attrition |
| 04 | Aggregate complete Monday–Sunday weeks and supporting coverage |
| 05 | Build calendar lags, labels, protected splits and coverage/sensitivity tables |
| 06 | Compare models, freeze selection/cutoff and report performance/limitations |

## Labels and research limits

A surge is a next-calendar-week observed local median increase of at least **10%**. One shared comparison handles tiny floating-point errors: RM5.50 → RM6.05 is correctly positive. Its tolerance is `1e-10` percentage points, not an economic margin. Preparation schema 3 records this rule; each `split_summary.csv` reports `numerical_boundary_corrections`. Historical schema-2 models retain their original strict labels and baseline rule during reuse; corrected labels require a separate new run.

Missing outcomes remain unlabelled. Exact-week lags, past-only rolling features, fold-specific preparation and training-only model preprocessing protect chronology. Three expanding validation folds are within 2025; holdout trains through December 2025, selects its score cutoff on January–March 2026 and evaluates on April–June 2026. Future prices/coverage never enter the model feature list.

Suspicious source prices are not automatically confirmed errors. The six documented conservative exclusions and remaining flags are recorded; the dataset is not fully verified. Report sparse/excluded areas, scarce group positives, aggregation sensitivity, false alerts, missed surges and temporal variation. Weighted scores are uncalibrated. Combined-model baseline improvements do not isolate the value of history alone. Affordability, adoption and inequality benefits have not been measured. SHAP, separate anomaly detection and dashboard evaluation remain next-stage work.
