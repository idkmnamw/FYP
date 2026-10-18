# PriceCatcher FYP

Only the main workflow is kept here: notebooks **01–08**, `pricecatcher_config.py`, `prepare.py`, `requirements.txt`, this README and `.gitignore`. Generated data/models stay out of Git. No extra application modules are required.

## Install once

Use Python 3.11 or newer, from the FYP folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If Notebook 06 reports missing `nbformat`, its kernel is using a Python without the project packages. In that notebook, run `%pip install -r requirements.txt` in a temporary cell, then Restart Kernel → Run All. On macOS, if LightGBM reports missing `libomp.dylib`, run `brew install libomp` in Terminal.

For Jupyter, run `python -m jupyter lab` and select this environment as the kernel. Restore an older saved model using the package versions in its own `model_manifest.json`.

## Everyday use: Run All in 01 → 08

1. Open the FYP folder in VS Code and select the project's `.venv` Python kernel.
2. Open Notebook **01** and select **Run All**. It starts a new run automatically, copying a verified cached source when available.
3. Run **02 → 03 → 04 → 05** in order. Each uses the same remembered folder and prints its path.
4. Run **06**. It prepares any missing expanding-window folds, then compares models and displays results. No folder, environment-variable or mode edits are needed.
5. Run **07** for separate Isolation Forest anomaly review. It uses the same active run and preserves every file from 01–06.
6. Run **08** to verify both models, precompute SHAP and prepare dashboard inputs. Install the new SHAP dependency once if requested.
7. To view results again, Run All in **06**, **07** or **08**; existing models are reused and a matching explanation cache is loaded. To repeat the entire experiment, start again at **01**.

Wait for each notebook to finish before starting the next. Do not start a new run in 01 while another notebook/terminal run is executing. Remove any old extra setup cells that manually set `PRICECATCHER_DATA_DIR` or `PRICECATCHER_EXPERIMENT` before using this automatic workflow. Those variables remain supported for advanced/terminal runs and override automatic selection.

Previous experiments stay under `data/runs/run_<timestamp>/`. A small generated `data/active_run.txt` remembers the current run across kernels; do not edit it during a run. Original data/models stored directly under `data/` are also preserved. Clearing notebook outputs or restarting a kernel does not delete files. A completed run's 02–05 inputs remain protected; restart at 01 for a new reproduction.

A complete cached source avoids monthly downloads. Without a cache, 01 needs access to `storage.data.gov.my`. That download was blocked by this cloud's proxy (403); normal Mac access may work. Failed downloads stop execution rather than export partial data. Notebook 06 can take longer because it prepares folds and compares eight candidates; its cell outputs show progress.

The configured April–June 2026 test has already been inspected. Every rebuild is a **reproduction**, not fresh independent validation. Do not tune against reproduced test results. Keep the source snapshot, environment and run folder for results cited in the report. Existing saved outputs describe their last execution; new outputs appear after Run All.

## Optional terminal runner

```bash
python prepare.py --rerun-all
```

This runs **01–06** with fresh kernels (run 07 and 08 interactively afterward), saves executed notebook copies inside the new run folder, and remembers that folder for later interactive use. You do not need this command for everyday manual execution. `prepare.py` stays because 06 uses it internally for missing folds.

Advanced users can select a specific saved folder explicitly:

```bash
PRICECATCHER_DATA_DIR="/path/to/FYP/data/runs/run_<timestamp>" python prepare.py --model-mode auto
```

A partial saved run is never silently replaced. Reuse verifies model/selection/input hashes and follows the saved preparation settings. It can restore metadata only when the entire reconstructed settings file has the exact recorded original hash and every other frozen input passes; the changed JSON is backed up. Pending evaluation resumes with the frozen model/cutoff. Changed model/data files remain blocking errors.

The supplementary source-quality audit is optional. Missing or mismatched weekly audit data is recorded as **unavailable**, with the reason; stale supplementary audit tables are removed. Verified main results can still be viewed, and unavailable never means the audit passed. Model/input mismatches and incompatible saved experiments remain blocking errors. Boosting libraries are imported only when their models are needed; training the full comparison still requires all declared dependencies.

## What each notebook does

| Notebook | Purpose |
|---|---|
| 01 | Integrate sources, validate joins/units and preserve the source snapshot |
| 02 | Screen item coverage using training data only |
| 03 | Clean invalid records, flag suspicious prices and audit geographical attrition |
| 04 | Aggregate complete Monday–Sunday weeks and supporting coverage |
| 05 | Build calendar lags, labels, protected splits and coverage/sensitivity tables, including matched-premise training comparisons |
| 06 | Compare models, freeze selection/cutoff and report performance/limitations with supplementary paired week-bootstrap intervals |
| 07 | Fit/reuse a training-only Isolation Forest and export separate anomaly scores, group summaries and review candidates |
| 08 | Verify frozen forecasts/anomalies, precompute SHAP and save bounded explanation caches plus all supported latest dashboard rows |

## Labels and research limits

A surge is a next-calendar-week observed local median increase of at least **10%**. One shared comparison handles tiny floating-point errors: RM5.50 → RM6.05 is correctly positive. Its tolerance is `1e-10` percentage points, not an economic margin. Preparation schema 3 records this rule; each `split_summary.csv` reports `numerical_boundary_corrections`. Historical schema-2 models retain their original strict labels and baseline rule during reuse; corrected labels require a separate new run.

Missing outcomes remain unlabelled. Exact-week lags, past-only rolling features, fold-specific preparation and training-only model preprocessing protect chronology. Three expanding validation folds are within 2025; holdout trains through December 2025, selects its score cutoff on January–March 2026 and evaluates on April–June 2026. Future prices/coverage never enter the model feature list.

Suspicious source prices are not automatically confirmed errors. The six documented conservative exclusions and remaining flags are recorded; the dataset is not fully verified. Report sparse/excluded areas, scarce group positives, aggregation sensitivity, false alerts, missed surges and temporal variation. Weighted scores are uncalibrated. Combined-model baseline improvements do not isolate the value of history alone. Affordability, adoption and inequality benefits have not been measured. Notebook 07 adds anomaly review without claiming verified anomaly accuracy. Notebook 08 explains the frozen forecasting model; dashboard implementation and evaluation remain next-stage work.

## Read the supplementary evidence

Notebook 05 adds `training_matched_premise_sensitivity.csv`: compare next-week prices using the same premises in both weeks, with one weekly median per premise. It reports support and label disagreement for one-/two-premise minimum panels. This uses training outcomes only and does not change labels, features or eligibility. Generate it by starting at Notebook 01 and continuing through 05; do not replace a frozen run's prepared files.

Notebook 06 adds `supplementary/test_week_bootstrap.csv` and its settings JSON. It resamples whole weeks 300 times, pairing the frozen model with the current-change and training food-group baselines. Main files, choices and cutoffs stay unchanged. These are post-hoc descriptive intervals within a short test period, not new independent validation or evidence across years. Week resampling does not preserve dependence between successive weeks. Reuse mode can generate them without refitting. Existing notebook outputs describe the last local execution; newly added cells need execution before quoting their results.

No new Python modules are needed for these checks. Notebook 07 provides separate Isolation Forest anomaly review; Notebook 08 adds the SHAP dependency and an offline dashboard cache. Dashboard implementation and evaluation remain next-stage work.

## Notebook 07 outputs and interpretation

After a completed 06, Run All in `07_anomaly_detection.ipynb`. It requires no mode/path edits or new dependencies. It fits six percentage-change/relative-volatility features with training-only median imputation and a fixed Isolation Forest (150 trees, 256 samples per tree, 1% training review budget, seed 42). No labels, future prices, future coverage or geographical categories enter the detector. It uses the supported modelling rows retained by 05, not every raw observation or district.

Five files stay inside the current run's `07_anomalies/` folder: `isolation_forest.joblib`, `anomaly_manifest.json`, `anomaly_scores.parquet`, `anomaly_summary.csv` and `review_candidates.csv`. Scores cover train/validation/test/latest; the review CSV contains at most 20 flagged cases per period. The manifest records the formulas, threshold, package versions and frozen input hashes. Repeated Run All reuses the same detector and regenerates tables; original 01–06 files are not changed.

Higher anomaly scores mean more unusual patterns, not event probabilities or confirmed errors. The 1% contamination setting sets a training review budget; later flag rates may differ. Report support/flag rates by food group, state and district, and inspect representation. There is no independent anomaly ground truth, so surge labels cannot be used as anomaly labels or to claim anomaly accuracy. Do not remove flagged prices or retune 06 from this review. An empty latest input produces no forecasts, not stale replacements. Notebook 08 precomputes SHAP explanations for the frozen supervised forecasting model and prepares dashboard inputs.

## Notebook 08 explanations and dashboard cache

After completed 06 and 07, open `08_shap_and_dashboard_inputs.ipynb` and Run All. For first use, install the updated requirements in the notebook's kernel (`%pip install -r requirements.txt`), then restart it. The forecasting model and Isolation Forest are loaded and scored, never fitted; saved predictions and flags must match. No earlier files are changed. Keep the package versions recorded in the saved models when restoring an older run.

A fixed random sample of up to 200 training rows supports a descriptive feature summary. Up to 200 highest-score latest rows get local explanations; every supported latest row remains in the dashboard data with `has_cached_explanation` indicating availability. Linear SHAP uses a training-only background; tree models use training path/leaf references. Native XGBoost TreeSHAP preserves sparse-input semantics. One-hot and missing-indicator contributions are summed to their original feature names. Additivity and the link to the frozen model score are checked for every explained row.

Four artifacts stay in `08_explanations/`: `latest_dashboard.parquet`, `shap_explanations.parquet`, `training_feature_importance.csv` and `explanation_manifest.json`. A matching cache is reused without SHAP computation; missing/changed derived cache files can be regenerated after upstream verification. The manifest records hashes, sample settings, output units, versions, coverage and offline build time. The notebook separately measures a local cache-file read; this does not prove dashboard/browser responsiveness.

SHAP values use model log-odds for logistic/boosting models and uncalibrated model-score units for Random Forest. They are associations conditional on the model/reference, not causes, verified event probabilities or evidence of affordability gains. A grouped one-hot display is not a separate jointly grouped SHAP game. The prioritised latest subset is not representative global importance. Display the historical data week, unsupported areas and unavailable explanations explicitly. Next: build and evaluate the dashboard using these cached files, without retuning 06 from exposed outcomes.
