# PriceCatcher FYP

Notebooks **01–06** prepare local weekly prices, compare surge classifiers and report their limitations. The code stays in the notebooks. There are only **two supporting Python files at the top level**:

| File/folder | Purpose |
|---|---|
| `pricecatcher_config.py` | Shared dates, item scope, coverage rules and output paths |
| `prepare.py` | Optional runner for 01–05; not needed for interactive notebook use |
| `requirements.txt` | Dependencies |
| `tests/` | Automated synthetic-data checks; not application code |
| `reports/` | Original result snapshot, qualified interpretation and verification environment |
| `data/` | Local source snapshots, prepared folds, saved model and results; excluded from Git |

The former `pricecatcher_features.py` and `pricecatcher_modelling.py` are no longer needed. Their necessary functions are inside the relevant notebooks. Notebook 90 remains optional source-review evidence.

## Your completed experiment: use this workflow

Keep your original `data/` folder and Python environment. Your trained model and Parquet files are local artifacts, not included in the GitHub notebook outputs. Use the package versions recorded in `data/06_models/model_manifest.json` when restoring an environment.

On your Mac, from the repository folder:

```bash
source .venv/bin/activate
python -m jupyter lab
```

Open **06_model_training_and_evaluation.ipynb**. Keep `RUN_MODE = "reuse"` and choose **Restart Kernel and Run All**, or execute it from the same terminal:

```bash
python prepare.py --model-mode reuse
```

This command does not regenerate prepared inputs. It:

1. Checks the saved model, selection and prepared-input hashes **before** checking legacy source-manifest consistency.
2. Loads the original model/cutoff and displays the main results without fitting or overwriting them.
3. Writes only the new supplementary results under `data/06_models/supplementary/`.

If the model or original data is missing, restore your original `data/06_models/` and `data/experiments/` folders. A notebook output cannot reconstruct a trained model. Do not switch to training mode merely to replace missing artifacts after inspecting the final test. The original notebook results are preserved in `reports/` and Git history.

## First run for a new study only

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare.py --experiment all
python -m jupyter lab
```

Run 01–05 manually in order if you prefer; select the experiment in `pricecatcher_config.py`, restarting kernels when changing it. Notebook 01 integrates the shared January 2025–June 2026 sources once. For each experiment, rerun only 02–05. Integration needs internet access and enough memory for the selected basket across 18 monthly files.

The runner reuses complete preparation by default, so repeating the command does not regenerate prepared files. Reproduction folders named `data_*` are also excluded from Git. If a notebook fails, its executed copy retains completed cells and the error for inspection; a failed run is never reported as completed. `--refresh` regenerates a study without a saved model. It refuses to overwrite preparation when `06_models/best_model.joblib` exists. For a new study, set a separate `PRICECATCHER_DATA_DIR` before starting Python/Jupyter; do not reuse the exposed test period as a fresh final test.

For an initial model experiment without existing saved artifacts, set `RUN_MODE = "train"` in 06 and run all cells. To run preparation and modelling together in a **separate** reproduction folder:

```bash
PRICECATCHER_DATA_DIR="$PWD/data_reproduction" python prepare.py --experiment all --model-mode train
```

On Windows set the environment variable using your shell's syntax. This reproduces the existing protocol; the already exposed test period does not become a new independent test. The executed Notebook 06 is saved inside its supplementary/executed folder. Training refuses to overwrite an existing model. Keep the first completed run and its artifacts.

## Notebook responsibilities

| Notebook | Main task |
|---|---|
| 01 | Integrate monthly sources/lookups, check joins/keys and standardise egg units |
| 02 | Screen item eligibility using the active training period only |
| 03 | Learn training-only price bounds, document cleaning and geographical attrition |
| 04 | Build Monday–Sunday prices, counts and equal-premise comparator |
| 05 | Build exact-week lags, fixed labels, protected splits and coverage audits |
| 06 | Compare four model families, freeze selection/cutoff, evaluate and interpret results |
| 90, optional | Review extreme training changes, common shops and documented exclusions |

Dates and output folders:

| Experiment | Training outcomes through | Validation outcomes through | Test outcomes through |
|---|---|---|---|
| fold_1 | 31 May 2025 | 31 July 2025 | — |
| fold_2 | 31 July 2025 | 30 September 2025 | — |
| fold_3 | 30 September 2025 | 31 December 2025 | — |
| holdout | 31 December 2025 | 31 March 2026 | 30 June 2026 |

```text
data/
  01_integrated/                 # shared integrated data, lookups and manifest
  experiments/
    fold_1/                      # fold_2, fold_3 and holdout use the same layout
      02_quality/
      03_cleaned/
      04_weekly/
      05_modelling/
      executed/                  # generated by prepare.py
  06_models/                     # original fitted model, cutoff and main metrics
    supplementary/              # new post-hoc checks; main artifacts stay unchanged
```

## Definitions and safeguards

A surge is an increase of **at least 10% in next week's observed local median**, for the same item, state, district and premise type:

```text
change_pct = 100 * (next_week_median / current_week_median - 1)
y = 1 if change_pct >= 10, otherwise 0
```

Unsupported pairs remain unlabelled. The main rule requires two records and one premise in both weeks, a previous-week price and two historical prices in the preceding four calendar weeks. Missing weeks are inserted temporarily for exact lags; prices/labels are not imputed. Partial boundary weeks and outcome horizons crossing split cutoffs are excluded. Features exclude future prices, coverage and labels.

Item screening, statistical bounds, model imputation, encoding and weights use each training period only. Full scope does not guarantee sufficient history everywhere. Use daily geographical attrition in 03, state/district eligibility tables in 05 and `series_coverage.csv` to report usable history and excluded places. Weekly exclusion reasons are sequential, not independent counts; final outcomes outside scope are structural rather than local missing-data failures. Completely unobserved places/weeks cannot be counted from observed records alone.

New source manifests record SHA-256 hashes of the integrated data and lookup files, combined into a content-based `source_snapshot_sha256`. Changing an integration timestamp alone does not change this content identity. New training checks content IDs when available and rejects inconsistent sources. The exact manifest hash is still retained for provenance.

In reuse mode, the original saved-model/input checks always run first. A legacy run with differing manifest hashes may then be inspected with `historical_source_consistency_unresolved` recorded in `supplementary/source_provenance.json`. This is not proof of identical source data or a retrospective repair of the original evaluation. Changed frozen inputs still stop execution; restore their original matching files rather than editing stored hashes.

Statistical flags are retained by default. The holdout training period excludes six documented suspected tenfold-price records without inventing corrections. Early folds retain these retrospective exceptions because their review dates were not recorded. Other suspicious prices remain uncertainty. Lookup metadata is a downloaded snapshot, and upstream monthly prices may be revised; preserve your integrated source/lookup files and manifest.

## Main and supplementary evidence in 06

The main experiment is unchanged: eight candidates across Logistic Regression, Random Forest, XGBoost and LightGBM; selection by mean fold average precision; final fit through December 2025; F1-based score cutoff chosen on January–March 2026; frozen evaluation on April–June 2026. The model is not refitted on validation after cutoff selection. `average_precision_score` is used, not trapezoidal PR-AUC. Scores from weighted models are uncalibrated. Undefined group metrics remain missing and support counts accompany the results.

Three original success checks compare mean fold AP, test AP and test F1 with the specified baselines. They are not evidence of universal superiority or demonstrated affordability benefits. Existing test results are exposed: further model development needs an explicitly exploratory label and an untouched period for a new final claim.

New supplementary checks are labelled **post-hoc**, outside the original benchmark selection:

- **Food-group prevalence baseline:** training-only group frequencies, global-training fallback for unseen groups, its own validation-selected cutoff. Tests whether ranking adds information beyond food-group composition.
- **Validation source-quality audit:** frozen predictions compared for pairs with/without suspicious records, with group counts. This is diagnostic evidence, not a causal explanation or a model-retraining robustness experiment. Target-week flags are audit evidence, never features.
- **Error and temporal context:** false alerts among alerts, false positive rate, missed-surge fraction, prevalence/support and AP context across periods.

Main files include `best_model.joblib`, `selection.json`, `model_manifest.json`, fold/validation/test metrics, group metrics and prediction Parquets. Supplementary outputs have their own settings file linking them to the original model and weekly audit. Later SHAP work should explain the saved pipeline and precompute selected explanations; Isolation Forest remains separate.

## Tests and report updates

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic monthly sources and real Parquet/Jupyter execution. The full command-line integration check runs 02–06 and optional 90 in fresh kernels across every fold; 01 is exercised against the synthetic monthly source fixture. It also verifies that diagnostics are not redundantly rerun for the holdout. They cover expanding-fold preparation, future-price independence, calendar gaps, training-only imputation, unseen categories, supplementary baseline fitting, saved-model prediction reproduction, hash checks and preservation of original artifacts during reuse. These tests do not establish real-data accuracy or robustness.

Update the report's older 1,950-record removal policy, boundary-week description and screenshots to match the actual run. Report baseline comparisons, false alerts, missed surges, food-group support, temporal variability and unresolved source uncertainty. Stakeholder adoption, affordability effects and omitted external drivers need appropriately limited claims; they are not proven by a good classification score.

## Cloud verification status — 17 October 2026

The cloud Python environment was restored and current dependencies were installed. Synthetic tests and real Jupyter executions are covered by the test suite, including unchanged frozen outputs and legacy snapshot handling. The attempted **full-source** run stopped at the first data download with `CONNECT tunnel failed: 403 Forbidden`: the cloud allowlist did not include `storage.data.gov.my`.

A cloud configuration draft adds that official host and saves installation/startup instructions. It must be reviewed, saved and published in Environment Settings before the real download can run. This changes cloud access, not the repository's data or model. No fresh full-data metrics are claimed while that blocker remains. See `reports/verification_status.md` for the distinction between historical results, synthetic verification and the pending real-data reproduction.

## Before the next FYP stage

- Keep the completed model/cutoff and original data artifacts. Use reuse mode for supplementary work.
- Finish one consistent-source real-data reproduction once the source host is accessible; a test suite alone is not that reproduction.
- Update the report's definition, preparation policies, baseline comparisons and limitations from the appropriate run, clearly distinguishing the original and supplementary analyses.
- Then explain the saved pipeline with SHAP, implement separate anomaly detection and evaluate the dashboard. Do not interpret weighted model scores as calibrated probabilities.

A supplementary AP calculation from archived group totals gives approximately **0.22529** for the training-food-group-prevalence baseline, versus the original selected model's reported **0.331978**. This checks information beyond food-group membership; it does not isolate price history from other model features. It is an aggregate calculation from saved tables, not a new full-data run or validation-cutoff result. See the verification report for assumptions and limitations.
