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
| **Research 12** Frozen test evaluation | Published deployment, saved test split and matching training-history artifacts | Retrospective test predictions, metrics, baseline comparisons and label audit; no model selection |
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
| 12 | Predict and evaluate the frozen dashboard model on the existing April–June 2026 test set; no fitting or dashboard updates |
| 13 | Optional fixed-configuration shop-composition experiment; requires the pinned longer-history tuned LightGBM, verified 2024 source and frozen fold inputs; six research fits, no deployment updates |

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

For test-set predictions, open [research/12_test_set_evaluation.ipynb](research/12_test_set_evaluation.ipynb) in the matching saved Python environment on your Mac and **Restart Kernel and Run All**. It uses the existing deployment pin, model, cutoff and `test.parquet`; there are no new dates or protocol modes to configure. It displays and saves predictions, overall metrics, a confusion matrix, weekly metrics and food-group metrics under the active run's `research/12_test_set_evaluation/<evaluation-id>/`. Existing main notebooks, models, dashboard exports and registrations are unchanged. Results must be described as a retrospective evaluation on a previously exposed test period, not an untouched independent test. Missing/mismatched artifacts stop execution; no retraining or downloading is performed.

Research 12 also saves test surge prevalence, fixed no-surge/current-change baseline metrics, a history-matched training food-group-rate baseline (overall and by food group), and a limited target audit using consecutive observed test prices. The food-group reference uses 2025 training for ordinary models or the selected longer-history model’s recorded 2024–2025 training support. It uses training rates only, falls back to the overall training rate for unseen groups, and reports ranking AP without choosing a classification cutoff. `training_food_group_rates.csv` records the rates and support. Longer-history evaluation requires the existing verified 2024 cache and cleaning artifacts; missing files stop execution without downloading or substituting 2025-only rates. It reports checked and unchecked label counts and stops on inconsistencies. A revised calculation creates a separate fingerprinted result folder, preserving earlier evaluations. These additions do not retrain the model or change its cutoff.

The report workflow is **main 10: compare models → main 11: freeze deployment → main 12: publish dashboard/SHAP → research 12: evaluate the frozen model → report all test groups and limitations**. Main 13 refreshes newer observations separately; it is not required for test evaluation. Do not use test results to choose a replacement model or cutoff. The dashboard's **How it works → April–June test performance** reads completed local results only when their model, deployment, test-file hash and cutoff match the displayed snapshot; CSV hashes are checked. It never runs evaluation automatically.

For full local SHAP coverage of the snapshot already on your dashboard, open main **12_shap_and_dashboard_inputs.ipynb**, leave `EXPORT_MODE = "auto"`, and use **Restart Kernel → Run All**. It expands or verifies every supported latest row and prints the actual coverage (for example, 5,343 / 5,343). This keeps the saved observation week, prices, model scores, cutoffs and anomaly flags. Earlier caches are retained; no new price files or registered outcomes are opened. Restart Streamlit afterwards. A first export or newly pinned deployment uses the original export route automatically. Run main 13 only when you want newer observations, subject to its registration guard. `prepare.py --verify-checkpoint` remains an optional additional check.

Future runs of main 12 and 13 also explain **every supported latest case**, using batches of at most 128 rows (smaller when needed for the dense-memory guard). The historical 200-row sample remains only for the descriptive global SHAP summary, and the 100-row linear-model background is unchanged; neither caps the latest-case explanations. Actual compute time depends on the saved model and machine. Main 13 still enforces its source/registration guards. Older published snapshots remain usable with their actual partial-coverage counts until expanded. A new verification fingerprint preserves earlier research 12 results if export provenance changes; reproducing scores is not a new independent test.

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

For a case explanation, open **Understand a prediction** and choose an item/location. The observed price/change and the following week's surge signal have separate dates and meanings. A short SHAP summary names the largest saved contributions raising and lowering each model score; it does not infer economic causes. Saved explanations show their signed contributions; their availability is counted within your filters, and missing explanations remain unavailable. **How it works** shows overall SHAP contribution sizes for the saved reference sample, separately from individual predictions. Contributions describe the model, not economic causes or percentage-point changes in event probability.

To inspect anomaly flags, select **Unusual patterns** in the local price watchlist, then choose a case under **Anomaly details**. The panel shows its saved anomaly score, the detector cutoff when verified against the export, recent price history, observation support and six price-change/variation measurements. Missing measurements remain blank; these are not anomaly feature attributions or proof of a source error. Classifier SHAP explanations concern the separate next-week surge score. These explanation views read existing artifacts and do not require rerunning notebooks 12/13.

Bundled [DOSM district boundaries](assets/malaysia-districts.geojson) are the unmodified 160-feature [official file](https://github.com/dosm-malaysia/data-open/blob/21a78e98efd4cd9b022a27a1bf67d167076b7591/datasets/geodata/administrative_2_district.geojson), retrieved 28 October 2026, under the [Open Data License](assets/DOSM-OPEN-DATA-LICENSE.md). No official endorsement is implied.

To verify the current export on your Mac, check main 12's actual latest-row coverage count and the dashboard's **How it works → April–June test results** panel (shown directly on that tab). Confirm that prediction explanations load. These are local verification steps, not checks performed against real artifacts in the cloud workspace.

The **How it works** tab shows verified retrospective test metrics and an AP comparison chart directly, with confusion counts in an expander. Baseline CSV hashes, support counts and model AP must agree with the matching evaluation; older evaluations without baseline exports remain usable without invented comparison values. Detailed score and missing-price notes are grouped in a separate reading guide. These presentation changes require only a dashboard restart, not notebook reruns.

The opening snapshot headline identifies the state/food-group combination with the most review alerts **within active filters**, alongside its monitored-series count and observed dates. Ties are disclosed; counts are not a population-risk ranking. On Streamlit versions supporting selectable tabs, its button opens the price map without changing filters; older versions show a map pointer. The top three case cards summarise their two largest saved SHAP contributions by magnitude and direction. Missing explanations remain explicitly unavailable. These additions read saved results only and require no notebook reruns.

For the optional shop-composition check, open **research/13_shop_composition.ipynb** in the matching saved environment and Restart Kernel → Run All. It compares original features with three current/past shop-history features using the deployed tuned LightGBM configuration and matching 2024-added training on all three existing folds. Reconstruction follows each fold’s hash-verified saved preparation rather than the active holdout configuration; it uses a local settings view and leaves the project configuration unchanged. Missing previous-week comparisons remain missing; observed zero shop overlap stays zero. Labels and evaluation rows remain identical. The protocol is saved before fitting: mean AP must rise and no fold may decline by more than 0.01 (an exploratory rule, not statistical significance). Completed identical research results are verified and reused. It does not score January–June 2026, change the deployment, select a cutoff, run earlier notebooks or download data. Results are saved only under the active run’s `research/13_shop_composition/<fingerprint>/`; share `fold_comparison.csv`, `fold_summary.csv` and `feature_coverage.csv`. Real-data improvement is unverified until this notebook runs with your local artifacts.
