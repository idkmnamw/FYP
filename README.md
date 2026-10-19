# PriceCatcher FYP

The main workflow is notebooks **01–08**, with optional **09–13** for research experiments, a historical-data audit and future-evaluation registration, `pricecatcher_config.py`, `prepare.py`, `requirements.txt`, this README and `.gitignore`. Generated data/models stay out of Git. No extra application modules are required.

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
| 09 (optional) | Exclude shortlisted suspected training records in a separate validation-only sensitivity comparison; preserve the main model/test |
| 10 (optional) | Compare original and matched-shop history features on the three 2025 expanding folds, preserving labels, rows and the main experiment |
| 11 (optional) | Audit 2024 file availability, core schema and bounded samples before considering longer training history |
| 12 (optional) | Integrate the audited 2024 basket and compare longer training history on unchanged validation data |
| 13 (optional) | Freeze both model copies and a November evaluation protocol without reading later outcomes |

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

## Review suspicious training prices before another experiment

Notebook 03 ends with an independent **Review training source prices** section. For a completed experiment, run **only that final code cell**; it uses the active run without invoking the earlier preparation guard. Do not start a new run in 01 just to review saved evidence. Normal 01–08 execution also includes it.

The generated `experiments/holdout/03_cleaned/training_source_context.csv` combines training statistical flags and documented conservative exclusions, verifies them against the integrated snapshot, and adds same-shop/item previous/following training observations, lookup units and implied pre-conversion prices. The last column is reconstructed evidence, not an independently verified raw price. Future training neighbours support retrospective investigation only; they are not forecast features. Existing preparation, model choices, cutoffs and results stay unchanged.

Review the table in your research notes, citing case IDs and independent evidence. Correct only confirmed errors in a separate experiment; retain unresolved values. This export is regenerated on rerun, so do not store decisions by editing it. No review outcome is automatically applied. Cached real data is required locally; saved notebook previews alone cannot confirm individual source records.

The same review export now includes an investigation priority and monthly source URL. Priority 1 highlights tenfold-or-larger deviations from both nearby (within seven days), similar neighbours; priority 2 highlights a large deviation from either neighbour. These are retrospective review aids, not confirmed errors or automatic exclusions. Use source/unit evidence to investigate the shortlist and include lower-priority cases when assessing missed issues. Fetching the official source confirms what was recorded, not necessarily what the true price should have been. Changes to cleaning require a separately evaluated experiment.

## Optional Notebook 09: test sensitivity to suspicious training prices

After the final review cell in 03 has produced the prioritised table, Run All in `09_source_quality_sensitivity.ipynb`. There are no mode/path changes and no need to rerun 01–08. The table's additional priority-1 cases (58 in the reviewed real run) are suspected, not confirmed errors. This exploratory analysis excludes them only from training, reconstructs training aggregates/features/eligibility/labels, and clones the selected model with its existing parameters. Before excluding anything, the reconstruction must match the frozen original training data.

Both versions use exactly the original validation features, rows, labels and original score cutoff. Validation history is intentionally unchanged to isolate the training intervention. No new cutoff, hyperparameter search, early-fold exclusions, test data access or dashboard replacement occurs. The priority rule uses retrospective neighbouring training observations, so it is not a deployable online cleaning rule. Report changes to training support/labels together with metrics, and do not tune the review policy repeatedly against validation performance. Test-period evidence remains exposed; this is not independent confirmation.

Results stay in `data/runs/<active-run>/09_source_quality_sensitivity/<fingerprint>/`. The fingerprint records code, model, input and package provenance; a matching completed cache reuses results without fitting. Seven artifacts include the supplementary model, excluded-record evidence, training-impact table, validation metrics/group metrics/predictions and manifest. Main experiment files remain unchanged. If original inputs differ, restore the matching originals; if code or reviewed evidence changes, a separate fingerprint preserves the earlier sensitivity results.

Run this on the Mac with its cached daily data and matching model environment. Cloud synthetic checks do not establish real-data results. Share `training_impact.csv`, `validation_comparison.csv` and `validation_group_comparison.csv` for interpretation before deciding whether another experiment is warranted.

## Optional Notebook 10: matched-shop feature experiment

Run All in `10_matched_shop_feature_experiment.ipynb` after completed 06. Notebook 09 is not a prerequisite. This compares the selected estimator's original features against two additional features: current/previous-week price change on common shops, and common-shop count. Every fold uses its own original daily cleaning/item eligibility, exact-week matches, training-only preprocessing and identical original labels/validation rows. Missing matches are retained, not treated as zero price change. No retrospective 09 exclusions, new cutoff or hyperparameter search is applied. The fixed fold cutoff is 0.5, matching the original fold comparison, not the final holdout cutoff.

The original-feature fold AP must reproduce the saved 06 result before a comparison is accepted. The current daily aggregates must also match the frozen price/coverage features. This requires the exact source/preparation and package versions from the main run; do not regenerate frozen inputs to bypass a mismatch. No prepared test file or 2026 holdout evaluation is accessed. Main models, dashboard inputs, labels and prior results stay unchanged.

A matching completed cache reopens without fitting. Six generated artifacts stay under `10_matched_shop_experiment/<fingerprint>/`: fold comparison, fold summary, food-group comparison, matched-shop coverage, validation predictions and a manifest. No model is deployed or saved by this development experiment. Share the first four CSVs for interpretation.

These are reused development folds, and the model settings were already selected using them. Any gain is exploratory, not fresh validation or proof of superior test performance. Report individual folds, group effects and missing-match coverage; prefer the simpler model if benefits are negligible or inconsistent. Future independent claims require an untouched later period and a frozen proposed policy. Source/unit uncertainty, publication delays, uncalibrated scores and user usefulness remain separate issues.

## Optional Notebook 11: check 2024 data before downloading the year

Run All in `11_historical_data_audit.ipynb` on the Mac using the existing project kernel. It does not run earlier notebooks, change the training scope, alter models or read the prepared test file. It checks the 12 official monthly URLs and core Parquet schemas, then samples first/middle/last row groups from January, July and December. Standard-library HTTP range reads and the existing PyArrow dependency cap data payload at 33 MiB for the audit. A server that ignores ranges is rejected instead of triggering a full-file download. Missing files, blocked access, incomplete samples and unsupported types stay explicit.

Three small reports are regenerated under `data/11_historical_audit/`: `monthly_availability.csv`, `selected_item_sample.csv` and `audit_summary.json`. No raw monthly files are saved. Share these reports before extending 01's source window. A zero sample count does not establish that an item is absent; full-year coverage and historical units/geography are not verified. Available cached current-study lookups are checked and used only as reference mappings.

This cloud's official-storage access is blocked by its proxy (403), so actual 2024 compatibility remains unassessed here. The audit stops further requests after a 403 rather than repeatedly retrying blocked URLs. The Mac may have direct access. If evidence is promising, the next research step is a separate longer-history comparison with training-only eligibility/preprocessing and fixed common evaluation rows; a positive audit alone is not evidence of model improvement.

## Optional Notebook 12: compare an extra year of history

After successful 11, Run All in `12_longer_history_experiment.ipynb` on the Mac. Do not change 01's dates or rerun 01–10. The first execution downloads all 12 audited monthly versions using their ETags and sizes, applies the original unit conversion/lookup contracts, and caches only the selected 2024 basket with raw-month hashes. Full-year integration stops on duplicate keys, changed source identities or incomplete downloads. Source rows missing required local-series metadata are retained in the selected cache and audited in `historical_metadata_review.csv`; they are excluded only from supplementary training, with counts in `historical_cleaning.csv`. The audit distinguishes absent shop codes from known codes with incomplete required fields. Missing display-only descriptions do not by themselves exclude a localisable record. No metadata is guessed and the original frozen lookup is not edited. The current frozen lookup is a provisional historical interpretation, not independent verification of historical units or shop attributes.

The comparison keeps the original 2025 item eligibility, retail filtering, selected model settings, target, original cutoff and validation inputs fixed. Invalid 2024 prices are removed by the same finite-positive rule; other suspicious prices remain. Original 2025 conservative exclusions stay as prepared; the later 09 shortlist is not applied. Training features/labels/eligibility are rebuilt using 2024–2025. The original-history reconstruction must match the frozen original, and all original training rows/labels must survive the extension. Fitted preprocessing and balanced weighting use only the new training data. Keeping eligibility fixed isolates history in the original basket rather than optimising all preparation policies.

The source cache lives under `data/12_longer_history/2024_source/<source-id>/`. Experiment results live under the active run's `12_longer_history_experiment/<fingerprint>/`; matching completed runs reuse data/results without downloads or fitting. Eight result artifacts contain the supplementary model, training totals/group support, historical cleaning counts, validation metrics/group metrics/predictions and manifest. Original models, earlier results, dashboard inputs and the active-run pointer stay unchanged. No prepared test file is accessed or cutoff retuned.

Share `training_comparison.csv`, `training_group_support.csv`, `historical_cleaning.csv`, `validation_comparison.csv` and `validation_group_comparison.csv`. More history may help or hurt. This is post-hoc exploratory validation, not fresh confirmation; do not automatically replace the main model or tune repeatedly against these results. Historical metadata and source quality, changing regimes and uncalibrated scores remain limitations. Independent confirmation requires an untouched later period after freezing the proposed policy.

## Optional Notebook 13: reserve a future comparison before the December deadline

Exposure to July–September 2026 outcomes is uncertain, so do not describe that period as independently untouched. Run All in `13_register_future_evaluation.ipynb` before 1 November 2026 to register the original model and the latest completed compatible 12 candidate. The selection is by completion time, not metric ranking; both identities/hashes are printed. No price data, labels, predictions, fitting, source download or metrics are accessed. The active run and all original artifacts are preserved.

Three files stay in `13_registered_future_evaluation/2026-11-02_2026-11-29/`: two verified model copies and `registration.json`. The registration fixes the original cutoff, model/settings/eligibility and a preliminary four-week outcome window of 2–29 November. Forecast-origin weeks start 26 October through 16 November; their closing dates are 1, 8, 15 and 22 November. Earliest outcome evaluation is 30 November, once complete source records are available. First-time late registration is rejected; repeated execution verifies the original registration without replacing it.

Registration is preparation, not an evaluation implementation or a new performance result. An evaluation stage must verify the frozen copies, preserve the declared policy, audit source/coverage and score both on identical supported rows. Four weeks are short and rare-food-group results may be unevaluable. The registered plan does not prove live predictions or point-in-time publisher availability; later source revisions remain a limitation. Build the dashboard meanwhile and report the 12 improvement as exploratory evidence until a valid later comparison is completed.
