# Localised food price surge preparation

A readable PriceCatcher pipeline for an FYP on next-week local food price surge risk. Notebooks **01–05** prepare data; **06** implements model comparison, train-fitted preprocessing, performance benchmarks and final evaluation. Notebook 90 is optional source evidence.

## Install and run

Use Python 3.11 or newer. From this repository:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows instead: .venv\Scripts\activate
python -m pip install -r requirements.txt

# Prepare the three expanding folds and the separate holdout experiment:
python prepare.py --experiment all
# Or prepare just one:
python prepare.py --experiment holdout
# Add --diagnostics to also execute optional Notebook 90.
```

The runner downloads/integrates sources with 01 only if the integrated Parquet or manifest is missing, then runs 02–05 with fresh notebook kernels. Successful executed notebooks are saved under the experiment's `executed/` folder. Any failed cell stops that run; do not continue with old downstream outputs. To refresh the shared integration intentionally, run 01 yourself and then rerun every experiment.

For interactive use, run `jupyter lab`, select `EXPERIMENT` in `pricecatcher_config.py`, then run **01 → 05** using Restart Kernel and Run All. When switching experiments, reuse 01 and repeat **02 → 05**. An explicit `PRICECATCHER_EXPERIMENT` environment variable overrides the default name in the file.

Keep the notebooks and Python helpers together. Data defaults to `data/` beside the configuration file; set `PRICECATCHER_DATA_DIR` before starting Python/Jupyter to override it. Generated data and executed notebooks are ignored by Git.

## Experiments and output folders

| Experiment | Training outcomes through | Validation outcomes through | Test outcomes through |
|---|---|---|---|
| fold_1 | 31 May 2025 | 31 July 2025 | None |
| fold_2 | 31 July 2025 | 30 September 2025 | None |
| fold_3 | 30 September 2025 | 31 December 2025 | None |
| holdout | 31 December 2025 | 31 March 2026 | 30 June 2026 |

All experiments share the same source start in January 2025. Expanding folds support model comparisons within 2025. The holdout experiment supports final selection on January–March 2026 and one final test on April–June 2026. These are preparation exports, not trained or evaluated models. Use date-based folds across all local series; do not split the stacked data by arbitrary row positions.

```text
data/
  01_integrated/                    # shared source and lookup snapshots
  experiments/
    fold_1/                        # fold_2, fold_3 and holdout follow the same layout
      02_quality/                  # training-only item eligibility
      03_cleaned/                  # training-only bounds and cleaning evidence
      04_weekly/                   # weekly prices within this experiment's scope
      05_modelling/                # train/validation, settings and coverage audits
      executed/                    # notebook outputs from prepare.py
```

Only `holdout/05_modelling/` includes `test.parquet`. Every experiment has its own learned item exclusions, price bounds and metadata. Later data does not determine earlier folds' preparation. Settings and source-manifest checks reject mismatched stages; changing any preparation setting requires rerunning 02–05 for that experiment. Existing outputs from the previous repository version are not migrated or deleted.

## Notebook responsibilities

| Notebook | Purpose | Main output |
|---|---|---|
| 01 | Integrate sources, validate joins, normalise egg units | Integrated daily Parquet and source manifest |
| 02 | Describe active scope; screen item coverage using training only | Item exclusion table and preparation settings |
| 03 | Clean invalid/excluded records; learn bounds from this fold's training period | Daily Parquet, cleaning summary and daily geographical audit |
| 04 | Aggregate complete Monday–Sunday scope weeks | Weekly Parquet and settings |
| 05 | Build exact-week features, labels, protected splits and audits | Model handoff, prediction inputs and research tables |
| 06 | Compare four model families across expanding folds; freeze model/cutoff and evaluate test | Saved pipeline, metrics, predictions and manifest |
| 90 | Optional extreme-change, matched-shop and documented-exclusion diagnostics | Source evidence CSVs |

Notebook 01 reads all 18 monthly files (January 2025–June 2026); allow internet access and sufficient memory. Lookup snapshots are reused. Monthly upstream files can be revised: preserve your integrated Parquet, manifest and lookups for final experiments. The manifest identifies the integration run but is not a guarantee that upstream prices never change. Lookup metadata is a downloaded snapshot, not a historical as-of lookup; acknowledge this limitation.

## Label and sparse-data rules

For series g and its observed median P in week t:

```text
next_week_change_pct = 100 * (P[g, t+1] / P[g, t] - 1)
surge_label = 1 when next_week_change_pct >= 10%; otherwise 0
```

The threshold is a fixed research definition, not an official economic standard. An unsupported price pair has a missing label, not zero. Prediction happens after the current week's observations are available. This observed median is not a fixed-shop price index: shop composition, monitoring frequency and promotions can affect it.

The main rule requires at least two records and one premise in both weeks. Historical support requires the previous calendar week's price and two observed prices in the prior four calendar weeks. Missing calendar weeks are inserted temporarily for correct lags; no prices or labels are imputed. Lag/rolling features use available past observations only. Partial scope weeks and label horizons crossing a cutoff are excluded. Later weekly predictions can use historical observations available by prediction time, without refitting a model on validation/test labels.

Training-only comparisons evaluate 5%, 10%, 15% and training-p95 surge definitions, equal-shop aggregation, and a stronger coverage rule of two premises/three observed days in both weeks. These tables do not automatically change the main policy. Equal-shop weighting still allows changes in shop identities. Fix policy choices before final testing.

## Coverage audits

Use these together; no single table represents all missing data:

- `03_cleaned/daily_geographical_coverage.csv`: source, removed and retained daily counts by state/district, including places lost entirely in cleaning. Removal reasons reconcile to the source total.
- `05_modelling/state_coverage.csv` and `district_coverage.csv`: observed weekly rows, eligible rows and exclusion reasons by period and overall. Reasons are **sequential and mutually exclusive**, not independent cause counts.
- `05_modelling/series_coverage.csv`: every source local series, observed/usable weeks and missing weeks inside its observed span. Fully removed series have zero usable weeks. Fewer than ten usable weeks is a diagnostic, not a further filter.

Weekly reason order is: outcome outside scope → horizon crossing cutoff → current coverage → missing next week → next-week coverage → insufficient history → eligible. The final week with no observable future is structural, not a local missing-data failure. Eligibility percentages exclude structural boundary rows from their denominator. Weekly audits cannot enumerate places/weeks never observed in the source. Descriptive test-coverage results must not be used to tune settings.

## Cleaning policy and metadata

Item eligibility and statistical bounds use only the active training period. Borong is excluded for retail scope. Statistical flags are retained by default; a flag does not prove an error.

The supplied research notebooks documented six suspected tenfold-price records. The holdout experiment excludes those exact training records after checking their keys/prices; no replacement prices are invented. Review dates were not recorded, so expanding folds **retain these manual exceptions by default** to avoid assuming earlier knowledge of a retrospective review. This is explicit in each experiment's metadata. If these source records are revised, the holdout stops for review. Notebook 90 checks the exclusions' training-label sensitivity when exclusions exist; it skips that comparison when none exist or optional statistical removal is enabled. Remaining suspicious prices are unresolved source uncertainty, not a reason to approve every outlier before modelling.

`PREPARATION_SETTINGS` records all active preparation choices and is linked to the integration manifest. The cleaning summary records actual removed counts, excluded item codes and learned bounds. Notebook 05 copies the actual cleaning policy and shared `TARGET_STATUS` into `experiment_settings.json`, alongside feature lists, split dates and summaries.

The report describes an older policy removing 1,950 flagged records and retaining partial boundary weeks. Update the methodology, screenshots, equation and counts from one final consistent run.

## Handoff to Notebook 06

Use each experiment's `experiment_settings.json` → `feature_columns` explicitly to build X. Treat item codes as categories. Fit imputation, encoding, scaling and any imbalance handling separately on each fold's training data. Never fit them on combined train/validation data. Do not impute missing future prices to manufacture labels. Use validation for classifier probability-cutoff selection; this is separate from the fixed 10% label threshold.

Compare Logistic Regression, Random Forest, XGBoost and LightGBM with modest tuning, an always-no-surge baseline and a simple current-change rule. Specify measurable success criteria before evaluation; report precision, recall, F1 and an explicitly defined precision–recall metric (such as average precision), fold variability and food-group support/results. Overall scores may be dominated by vegetables. Report undefined metrics for groups without the required positive/negative cases rather than inventing examples.

`latest_prediction_inputs.parquet` contains eligible inputs for the last complete week of the active experiment, without labels or future observations. It is a historical prototype snapshot, not a live feed. Empty output is reported instead of substituting stale weeks. Later dashboard work should precompute selected SHAP explanations and keep Isolation Forest anomaly scores separate from surge labels.

These preparation changes do not establish affordability improvements, stakeholder adoption or superiority over existing systems. Those require appropriate evaluation and careful report claims. Missing external drivers remain a documented limitation.

## Validation

```bash
python -m unittest discover -s tests -v
```

Synthetic tests exercise 01–05, all three folds and the holdout, real Parquet handoffs, optional diagnostics, date boundaries, calendar gaps, geographical reconciliation and stale-setting rejection. A future-price perturbation checks that an early fold's item eligibility, learned bounds and training export stay unchanged. Tests need no network or real research data. Notebook 06 is provided without run results; existing preparation notebooks may contain saved outputs. Run against your own preserved source snapshot before reporting final counts or claiming model performance.

## Notebook 06: model training and final evaluation

After installing the updated requirements and preparing **all** experiments, open `06_model_training_and_evaluation.ipynb` in Jupyter and choose **Restart Kernel and Run All**. `prepare.py` still prepares 01–05; it deliberately does not train models. Notebook 06 reads all four experiment folders independently of the current interactive `EXPERIMENT` selection.

The notebook compares two modest settings each for Logistic Regression, Random Forest, XGBoost and LightGBM. It uses a fresh training-fitted median imputer, missingness indicators and one-hot encoder for every fit, scaling numeric inputs for Logistic Regression only. Class weights come from training data; no SMOTE or combined train/validation preprocessing is used. CPU use defaults to two jobs and tree ensembles to 150 iterations/trees. The full run requires 24 expanding-fold fits plus one final fit; runtime depends on the real dataset and your hardware. Linux installs the CPU-only XGBoost distribution to avoid unnecessary GPU dependencies.

Selection follows this order:

1. Compare the eight candidates using mean average precision across the three 2025 folds. Report fold standard deviation and precision/recall/F1 at the fixed 0.5 score cutoff.
2. Freeze the strongest ML candidate. Fit it on the holdout training data through December 2025.
3. Choose its score cutoff by maximum F1 on January–March 2026 validation data; ties use the highest cutoff. This validation F1 is a selection result, not a final performance estimate.
4. Save the fitted pipeline and selection before reading test data. Evaluate the frozen choice once on April–June 2026. Do not change settings after viewing this result and continue calling that same period an untouched test.

The model is not refitted on validation after cutoff selection. Weighted models' scores are not calibrated probabilities. Average precision is `sklearn.metrics.average_precision_score`, **not trapezoidal PR-AUC**; use that exact definition in the report. Always-no-surge and current-change-rule baselines are reported on the same rows. AP without both classes and recall/F1 without observed positives remain undefined. The notebook stops if the required training/fold-validation support cannot sustain a fair comparison.

The three prespecified success checks are mean fold AP above both baselines, test AP above both baselines, and test F1 above both baselines. These relative benchmarks can fail, and the notebook reports that honestly. They do not claim a universally best framework, stakeholder adoption or demonstrated affordability effects.

Outputs are under `data/06_models/`:

| Output | Use |
|---|---|
| `fold_metrics.csv`, `fold_summary.csv` | Candidate comparisons and variability |
| `validation_metrics.csv`, `selection.json` | Validation cutoff and frozen selection |
| `best_model.joblib` | The complete fitted preprocessing + model pipeline |
| `test_metrics.csv`, `benchmark_checks.csv` | Final comparison with explicit success checks |
| `test_group_metrics.csv`, `test_evaluation.png` | Food-group/geographical support, errors and PR curves |
| `validation_predictions.parquet`, `test_predictions.parquet` | Auditable identifiers, labels, scores and decisions |
| `latest_predictions.parquet` | Historical-snapshot predictions without outcome labels |
| `model_manifest.json` | Candidate settings, package versions, preparation policy and input/code/model hashes |

Notebook 06 saves outputs when rerun, so preserve the final run and its source/preparation artifacts. Later SHAP work should explain this saved pipeline and precompute selected explanations; anomaly detection remains separate. Those later components are not implemented by Notebook 06.

The tests include a real-kernel execution of the complete notebook against synthetic prepared folds, all eight candidates, saved-model prediction reproduction, training-only imputation, unseen categories, unsupported-class metrics and validation cutoff selection. Synthetic scores do not represent FYP performance: execute the real pipeline before reporting findings.
