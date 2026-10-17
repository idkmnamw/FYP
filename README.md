# PriceCatcher FYP

Only the main workflow is kept here:

| Files | Purpose |
|---|---|
| Notebooks 01–05 | Integrate, clean, aggregate and prepare fold-safe features/labels |
| Notebook 06 | Compare models and evaluate next-week surges |
| `pricecatcher_config.py` | Shared settings, output paths and source checks |
| `prepare.py` | Optional runner for the same notebooks |
| `requirements.txt` | Dependencies |
| `.gitignore` | Keeps generated data/models and caches out of Git |

## Run Notebook 06

Keep `RUN_MODE = "auto"`, then **Restart Kernel → Run All**.

- If a saved model/manifest exists, 06 verifies and reuses that run.
- If neither exists, 06 runs the fixed experiment into `data/06_models_reproduction/`. It can prepare missing folds from the integrated local data produced by 01. The original model folder is preserved.
- A partially saved or changed run stops with the missing/changed file path. It is never silently replaced.
- A recoverable holdout settings mismatch is repaired only if the complete reconstructed file matches its originally recorded hash and all other frozen artifacts pass. The changed JSON is backed up locally. Model/Parquet files and expected hashes are preserved.

If both `best_model.joblib` and `model_manifest.json` are missing, there is no saved model to reuse. Auto handles that case as a separate reproduction. Existing complete local saved files can be reused without any source download.

The April–June 2026 test has already been inspected. Rebuilding the experiment is a **reproduction**, not fresh independent validation. Do not tune against its test results or replace the original reported cutoff because another cutoff performs better on test.

## Install and run from a terminal

Use Python 3.11 or newer, from this repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare.py --model-mode auto
```

Use the original model manifest's package versions when restoring its environment. For Jupyter, run `python -m jupyter lab` and select that environment as the kernel.

For a new full reproduction in a separate folder:

```bash
PRICECATCHER_DATA_DIR="$PWD/data_reproduction" python prepare.py --experiment all --model-mode train
```

The runner saves executed notebook copies under `data/`. If a cell fails, the executed copy retains the error. Actual source preparation needs internet access to `storage.data.gov.my`; the cloud proxy currently returns 403. This does not block reuse of complete local artifacts. Direct preparation stops when either model folder contains a frozen model; use a separate data folder for changed preparation.

## Research rules

A surge is a next-calendar-week local median increase of at least 10%. Missing outcomes remain unlabelled. Exact-week lags, past-only rolling features, fold-specific preparation and training-only model preprocessing protect chronology. Validation selects the score cutoff; test does not.

Report AP, F1, precision, recall, confusion counts, group/geographical support and temporal variability. Weighted scores are uncalibrated. The combined model includes history and non-history inputs, so baseline improvements do not isolate the effect of price history. Affordability, adoption and inequality benefits have not been demonstrated. Keep SHAP, separate anomaly detection and dashboard evaluation as the next stage.
