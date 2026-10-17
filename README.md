# PriceCatcher FYP

Notebooks 01–06 prepare weekly local prices, compare next-week surge classifiers and report their limitations. Code stays in the notebooks, with only two supporting Python files.

| File/folder | Purpose |
|---|---|
| `01_…ipynb`–`06_…ipynb` | Main preparation and modelling workflow |
| `90_optional_source_diagnostics.ipynb` | Optional training-data review and label sensitivity |
| `pricecatcher_config.py` | Dates, item scope, coverage rules, paths and source verification |
| `prepare.py` | Optional notebook runner, including model training/reuse |
| `requirements.txt` | Installation dependencies |
| `tests/` | Synthetic regression and fresh-kernel execution checks |
| `reports/` | Historical results, current verification status and tested environment |
| `data/` | Generated local artifacts; excluded from Git |

Repository notebooks are source copies with saved outputs cleared. The runner saves executed copies under `data/`; historical real-data results remain in `reports/completed_main_experiment.md` and Git history. Synthetic outputs are never presented as FYP performance.

## Install

Use Python 3.11 or newer. In the FYP folder on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For an existing saved model, use its original environment/package versions from `data/06_models/model_manifest.json`. `reports/verification_environment.txt` records the cloud test environment, which can differ from the historical model environment.

## Existing completed experiment

Keep the original `data/06_models/` and `data/experiments/` folders. They contain local files that GitHub notebook outputs cannot recreate.

```bash
python prepare.py --model-mode reuse
```

Alternatively, open Jupyter with `python -m jupyter lab`, keep `RUN_MODE = "reuse"` in 06, then **Restart Kernel and Run All**. Reuse checks model, selection and prepared-input hashes before loading the model. For a completed run it preserves all main files and writes supplementary analyses separately.

If a prepared-input hash differs, restore the original matching file rather than editing the expected hash or retraining against an exposed test. A JSON-only mismatch may involve provenance/formatting; compare it with the saved manifest before changing data. The uploaded holdout settings were recoverable byte-for-byte from their original preparation metadata, but this does not prove that other local files are unchanged.

New runs save a frozen model, cutoff and input-hash checkpoint before loading test rows. If execution stops afterward, the same reuse command finishes pending evaluation/exports without fitting or choosing another cutoff. The manifest changes from `evaluation_pending` to `completed` only after main exports succeed. A checkpoint failure before model activation leaves training restartable before test access. Older incomplete runs need their original matching artifacts; missing metadata is not invented.

## Separate reproduction or new study

Use a separate output folder to preserve the completed experiment:

```bash
PRICECATCHER_DATA_DIR="$PWD/data_reproduction" python prepare.py --experiment all --model-mode train --diagnostics
```

This integrates once, runs 02–05 and optional 90 for every fold, then trains/evaluates 06. It requires HTTPS access to `storage.data.gov.my` and enough memory for the selected basket across 18 monthly files. On Windows set `PRICECATCHER_DATA_DIR` using your shell's syntax.

The existing April–June 2026 test is already exposed. Reproducing it does not create fresh independent evidence; new model development needs an exploratory label and an untouched period for a new final claim.

For preparation only, use `python prepare.py --experiment all`. The runner reuses complete preparation by default. `--refresh` regenerates preparation only when no frozen model exists. Direct runs of notebooks 01–05 also stop if a saved model is present, preventing accidental replacement of its inputs. To run manually, set the output-folder environment variable before starting Jupyter, choose the experiment in configuration, and restart kernels when changing it.

## Notebook order and outputs

| Notebook | Task |
|---|---|
| 01 | Integrate monthly data/lookups, validate keys/joins and standardise egg units |
| 02 | Screen items using the active training period only |
| 03 | Learn training-only bounds, document cleaning and geographical attrition |
| 04 | Aggregate complete Monday–Sunday weeks and audit their coverage |
| 05 | Build exact-week features, labels, chronological splits and geography audits |
| 06 | Compare models, freeze selection/cutoff, evaluate and interpret results |
| 90 | Optional extreme-change, matched-premise and exclusion-sensitivity diagnostics |

| Experiment | Training outcomes through | Validation outcomes through | Test outcomes through |
|---|---|---|---|
| fold_1 | 31 May 2025 | 31 July 2025 | — |
| fold_2 | 31 July 2025 | 30 September 2025 | — |
| fold_3 | 30 September 2025 | 31 December 2025 | — |
| holdout | 31 December 2025 | 31 March 2026 | 30 June 2026 |

```text
data/
  01_integrated/             # shared source, lookups, manifest and executed 01
  experiments/
    fold_1/                  # fold_2, fold_3 and holdout have the same layout
      02_quality/
      03_cleaned/
      04_weekly/
      05_modelling/
      executed/              # executed preparation/diagnostic notebooks
      jupyter/               # runner's local kernel specification
  06_models/                 # frozen model, main metrics and prediction files
    supplementary/           # post-hoc results and executed model notebook
```

A failed notebook retains its completed cells and error in the executed copy. Generated data, reproduction folders, kernels, models and virtual environments stay out of Git.

## Definitions and interpretation

A surge is an increase of **at least 10% in next week's observed local median** for the same item, state, district and premise type: `100 * (next_median / current_median - 1) >= 10`.

Historical labels require two records and one premise in both weeks. Model-ready rows require a prior-week price and two observations in the previous four calendar weeks. Unsupported outcomes stay unlabelled; prices are not filled. Exact-week lags and past-only rolling features preserve gaps. Scope-boundary weeks and outcomes crossing split cutoffs are excluded. Item screening, bounds, imputation, encoding and class weights use each training period only. Source files are checked against recorded content hashes before preparation.

Use 03's daily geographical attrition and 05's state/district/series tables to report sparse or excluded areas. Weekly reasons are sequential counts; completely unobserved places/weeks cannot be inferred from observed records. A weekly median reflects monitoring frequency and changing contributors, so it is not a fixed-shop price index. Statistical flags remain unconfirmed; the six documented suspected records are excluded only in holdout training by default, with sensitivity reported separately.

The main experiment compares eight candidates across Logistic Regression, Random Forest, XGBoost and LightGBM. Mean expanding-fold average precision selects the candidate; 2026 validation F1 selects its score cutoff. The model is not refitted on validation. AP uses sklearn's average precision, not trapezoidal PR-AUC. Weighted scores are uncalibrated. Close candidate scores do not establish universal superiority.

Report false alerts, missed surges, group support and temporal variation. Comparisons evaluate the combined history, item/location, coverage and seasonal model; they do not isolate the effect of price history. Affordability, adoption and inequality benefits were not measured. Correct the report's older removal policy and screenshots to match the actual experiment.

Supplementary analyses are explicitly post-hoc: a training-only food-group prevalence baseline, validation quality-flag diagnostics, and error/temporal context. They do not change model selection or the original cutoff. Legacy source inconsistencies are reported as unresolved after frozen artifact verification. A weekly audit with different provenance must still match preparation policies, current validation prices and target labels; the difference is recorded rather than treated as proof of identical historical sources or flags.

## Verification and next stage

```bash
python -m unittest discover -s tests -v
```

Tests exercise synthetic sources and fresh-kernel notebook execution, source integrity, calendar gaps, fold boundaries, coverage, training-only preprocessing, saved predictions, recovery and preservation of frozen artifacts. They do not establish full-data performance. See `reports/verification_status.md` for the latest result and real-data run status.

After a consistent-source real-data reproduction and honest report update, continue with SHAP explanations of the saved pipeline, separate anomaly detection and dashboard evaluation. Precompute selected explanations for responsiveness.
