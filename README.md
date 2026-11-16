# HargaLens — Malaysian food-price monitoring

HargaLens is an FYP research prototype using PriceCatcher to review local weekly prices, next-week surge scores, unusual patterns and saved explanations. The target is an **observed local median price rise of at least 10% in the next exact calendar week**. It is not a fixed-shop price index. Scores are uncalibrated; alerts need investigation.

## Start here

Use Python **3.12** from the FYP folder. For a fresh development installation:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m streamlit run app.py
```

Select `.venv312/bin/python` for notebook kernels in VS Code. Run notebooks from the project root. Do not run notebooks concurrently or edit `data/active_run.txt` during execution.

**A fresh clone has no real datasets, fitted models or dashboard exports.** Opening the dashboard does not require rerunning notebooks, but does require the matching saved artifacts below. Without them, the dashboard reports missing inputs. Do not retrain merely to reopen saved results.

`requirements.txt` contains broad version ranges, not a checkpoint lock. To restore a saved model, use its `environment_requirements.txt` with the matching Python version in a separate environment. Linux uses `xgboost-cpu`; macOS uses `xgboost`, and LightGBM may need `brew install libomp`.

## Main workflow and dependencies

| Stage | Reads | Produces / purpose |
|---|---|---|
| **01** Integrate | Monthly 2025–June 2026 PriceCatcher files and item/premise lookups, or verified source cache | A new run with frozen `01_integrated/` data, lookups and source hashes |
| **02** Quality | 01 and experiment cutoff | Training-only item coverage and sparse-item screening |
| **03** Cleaning | 01–02 | Daily prices, training-learned price flags and source-review evidence; flags are retained by default |
| **04** Weekly aggregation | 03 | Complete-scope local weekly medians and coverage |
| **05** Features and labels | 03–04 | Exact-week lags, chronological train/validation/test splits, latest inputs and settings under `experiments/` |
| **05b** Historical audit | 2024 source metadata/bounded samples; existing lookups if available | Shared availability audit; prerequisite for 09b |
| **06–09** Classifiers | 05 for holdout and three expanding 2025 folds | Separate Logistic Regression, Random Forest, XGBoost and LightGBM studies; 16 Optuna trials each |
| **09b** Longer-history LightGBM | Completed 09, 05b audit, matching cleaned inputs and verified 2024 cache/source | 2024-added reference and inherited-tuned controls, plus a 24-trial direct-history search; original evaluation rows stay fixed |
| **10** Comparison | Matching 06–09/09b studies and source caches | Verified metrics, history-matched food-group baselines and a candidate under the fold-AP rule |
| **11** Anomalies | 10 or existing deployment pin, original 2025 training features | Separate frozen Isolation Forest and scores; creates the classifier deployment pin only if absent |
| **12** Explanations/export | Pinned classifier, 11 detector, prepared history and latest inputs | SHAP cache and dashboard exports under the pinned deployment |
| **13** Optional refresh | Existing dashboard, frozen models/cleaning/lookups and later observations | Derived offline snapshot without retraining; blocks registered outcome months |

For a new reproduction, run **01 → 05 → 05b → 06–09 → 09b → 10**, then **11 → 12** only if dashboard exports are needed. After 05, the classifier notebooks prepare missing expanding folds when it is safe to do so. Preparation is blocked when frozen studies would be overwritten. Repeated studies create timestamped outputs; existing deployment pins are preserved.

Convenience commands (these can execute notebooks or write local outputs):

```bash
python prepare.py --experiment all       # Prepare all chronological splits
python prepare.py --rerun-all           # New reproduction through 10; no deployment
python prepare.py --export-environment  # Write installed direct-package versions
python prepare.py --verify-checkpoint   # Read-only saved model/score/SHAP check
```

Uncached downloads require access to `storage.data.gov.my`. Preserve TLS, source hashes and ETag checks. If a source or manifest differs, restore the matching snapshot or use a separate new run; do not change hashes to make verification pass.

## Restore and preserve saved work

The active run is named by `data/active_run.txt` as `runs/run_...`. Preserve these together:

- The **complete** active `data/runs/run_.../` folder, including prepared inputs, models, study/comparison manifests, deployment pointers, dashboard/refresh archives, research outputs and registered evaluations.
- `data/active_run.txt` and the active run's `environment_requirements.txt`.
- Shared `data/11_historical_audit/` and `data/12_longer_history/` source caches.
- Any older folders referenced by saved manifests or registrations, even if VS Code hides them.

New input references use `run::` and `repo::`; legacy paths are resolved while their original hashes remain unchanged. A current notebook can differ from code recorded in an older study. Retain the matching Git revision when reproducing that evidence.

`pricecatcher_config.py` defines shared paths/settings and provenance checks; `prepare.py` runs preparation and compatibility checks; `app.py` reads dashboard exports. Keep main notebooks in the root. `assets/` contains the logo, district boundaries and required licence. `.vscode/settings.json` only hides known generated/legacy folders; hiding is not evidence that deletion is safe. Do not remove environments, checkpoints, runs or exports without checking their references and preserving local work.

## Optional and legacy research

Everything under `research/` is outside the main run sequence. It must not silently replace the dashboard model.

Artifact folders retain their original names for compatibility: main Notebook 12 writes to `08_dashboard`, Notebook 13 uses `09_refresh`, and archived exports may use `07_anomalies`. Do not rename these folders without checking saved references. Nonsequential execution counters can reflect standalone diagnostic cells; they do not establish a fresh end-to-end run. Do not rerun output-writing notebooks solely to tidy counters.

The current deployment and registered models must be verified from real saved manifests; their identities and metrics cannot be inferred from earlier notes. Research 04 targets older checkpoints and contains registration code, not an evaluation runner. Keep any existing registration unchanged and outside current-model performance claims unless its exact scope and results are verified.

| Notebooks | Status / prerequisites |
|---|---|
| 01–03 | Historical source-quality, matched-shop and history experiments; require their original combined-model checkpoint and possibly its Git revision |
| 04 | Legacy future-evaluation registration tool; preserve existing registrations, never present a late registration as prospective |
| 05 | Section 2 is the standalone source-review log; other sections use an older checkpoint |
| 06–08 | Relative-price, peer-district and larger-search studies; require completed 09b outputs and matching caches |
| 09 | Label/anomaly audit; requires completed 05/11 |
| 10 | Wholesale availability audit; reads existing training caches, no downloads or fitting |
| 11 | Fixed-parameter festival-calendar experiment; requires completed 09b Optuna outputs |

Research notebooks are numbered independently from the main pipeline. Renaming does not change their prerequisites or make them a sequence to run in order. Saved data/output folders keep their original names.

| Old research number | Current file |
|---|---|
| 09 | [01_source_quality_sensitivity.ipynb](research/01_source_quality_sensitivity.ipynb) |
| 10 | [02_matched_shop_feature_experiment.ipynb](research/02_matched_shop_feature_experiment.ipynb) |
| 12 | [03_longer_history_experiment.ipynb](research/03_longer_history_experiment.ipynb) |
| 13 | [04_register_future_evaluation.ipynb](research/04_register_future_evaluation.ipynb) |
| 14 | [05_review_readiness.ipynb](research/05_review_readiness.ipynb) |
| 15 | [06_relative_price_features.ipynb](research/06_relative_price_features.ipynb) |
| 16 | [07_cross_district_signals.ipynb](research/07_cross_district_signals.ipynb) |
| 17 | [08_larger_optuna_budget.ipynb](research/08_larger_optuna_budget.ipynb) |
| 18 | [09_label_and_anomaly_audit.ipynb](research/09_label_and_anomaly_audit.ipynb) |
| 19 | [10_wholesale_availability_audit.ipynb](research/10_wholesale_availability_audit.ipynb) |
| 20 | [11_festival_calendar_experiment.ipynb](research/11_festival_calendar_experiment.ipynb) |

Old notebook names are resolved by `pricecatcher_config.py` for legacy callers and saved input references. Hash checks still apply: edited notebook source has a new hash, so exact historical reproduction may require the original Git revision. No saved hashes or model manifests were rewritten.

The research notebooks retain their saved results and methodological caveats. Saved outputs are historical records, not evidence of a fresh cloud execution.

## Evaluation and claims

- The original four-classifier comparison trains on **2025**. Longer-history LightGBM additionally uses **2024** training history; evaluation rows remain matched.
- Three expanding 2025 folds select parameters using average precision (AP). January–March 2026 selects score cutoffs, so its classification results are apparent cutoff-selection results. Previously inspected validation/test periods are **not untouched evaluations**. Older April–June results do not transfer to a newer candidate.
- Read the current `11_deployment/deployment_selection.json` and verify its referenced study/comparison hashes before naming the deployed model. An earlier record of “Longer-history LightGBM (tuned)” does not establish the current selection. Later comparisons do not replace the existing pin.
- Preserve November registrations and their exact rules. Registered outcome months are blocked by refresh. A retrospective refresh is not a timely archived forecast.
- Sparse coverage, changing shops, provisional historical metadata and unresolved suspicious source cases limit interpretation. SHAP describes model contributions, not economic causes. Savings, affordability benefits and stakeholder adoption have not been demonstrated.

## Checks and troubleshooting

Check installed dependency consistency without executing notebooks or fitting models:

```bash
python -m pip check
```

`--verify-checkpoint` requires the actual saved deployment and exports. Missing artifacts cannot establish model compatibility, accuracy or a working real-data dashboard. Dependency checks and syntax validation do not establish end-to-end correctness. This checkout does not currently include a standalone test suite.

If dashboard inputs are missing, restore the matching snapshot first. With the required saved inputs present, 11 → 12 builds exports; 13 verifies/publishes a separate refresh. Do not restart 01 just to view results.

For the map, select an item/unit and shop type, then move or play the weekly timeline. Gaps remain gaps; grey means insufficient comparable data. District selection pauses playback; reset District to **All** for wider comparisons. Sampled medians do not guarantee the cheapest shop or population coverage.

To inspect anomaly flags, select **Unusual patterns** in the local price watchlist, then choose a case under **Anomaly details**. The panel shows its saved anomaly score, the detector cutoff when verified against the export, recent price history, observation support and six price-change/variation measurements. Missing measurements remain blank; these are not anomaly feature attributions or proof of a source error. Classifier SHAP explanations concern the separate next-week surge score. This view reads existing artifacts and does not require rerunning notebooks 12/13.

Bundled [DOSM district boundaries](assets/malaysia-districts.geojson) are the unmodified 160-feature [official file](https://github.com/dosm-malaysia/data-open/blob/21a78e98efd4cd9b022a27a1bf67d167076b7591/datasets/geodata/administrative_2_district.geojson), retrieved 28 October 2026, under the [Open Data License](assets/DOSM-OPEN-DATA-LICENSE.md). No official endorsement is implied.
