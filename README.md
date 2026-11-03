# HargaLens — Malaysian food-price monitoring

Weekly local prices, next-week surge review, unusual patterns and precomputed explanations. This is a research prototype: alerts require investigation, model scores are uncalibrated, and affordability benefits have not been demonstrated.

## Open the dashboard

Use the verified Python 3.12 environment from the FYP folder:

```bash
source .venv312/bin/activate
python -m streamlit run app.py
```

Select the same `.venv312/bin/python` interpreter for notebook kernels in VS Code. Opening the dashboard does not require rerunning notebooks. Its three tabs show saved prices, district maps with weekly playback, and explanations of the system.

## Project layout

| Files | Purpose |
|---|---|
| 01–05 notebooks | Integrate, audit, clean, aggregate and create features/labels |
| 05b | Audit 2024 source availability before adding history |
| 06–09 | Separate Logistic Regression, Random Forest, XGBoost and LightGBM studies |
| 09b | Longer-history LightGBM controls and a focused 24-trial Optuna search |
| 10 | Compare candidates and history-matched food-group-rate baselines |
| 11–12 | Pin a model, detect anomalies, precompute SHAP and publish dashboard inputs |
| 13 | Refresh observations with the existing model and cutoff |
| `app.py` | Read-only Streamlit dashboard |
| `pricecatcher_config.py` | Shared settings, paths and provenance checks |
| `prepare.py` | Prepare folds, reproduce studies and check the saved environment/model |
| `research/` | Supporting experiments and historical evidence; outside the main run sequence |
| `assets/` | Dashboard logo, district boundaries and their required licence |
| `data/` | Local source caches, experiments, models and exports; excluded from Git |

There are **three Python files**. Keep notebooks in the root: the runner and shared calculations use those paths. The VS Code settings hide virtual environments, Python caches and notebook checkpoints; this only changes the Explorer view. It does not delete files or select the notebook kernel.

**Keep `data/`, `.venv312`, `research/` and `assets/`.** `.venv` is the older environment on your Mac; remove it locally only after confirming no terminal, notebook or other project uses it. `__pycache__/` and `.ipynb_checkpoints/` are generated folders; checkpoints may contain unsaved work. Do not delete saved runs or manifests simply to tidy the Explorer.

## What to run

- **View the dashboard:** use the command above.
- **Review saved results:** open the notebooks; no execution is needed.
- **Repeat longer-history tuning:** Restart Kernel and Run All in **09b → 10** using `.venv312`. Keep existing preparation and 06–09 results.
- **Reproduce the full project:** run **01 → 05 → 05b → 06–09 → 09b → 10**. Notebook 01 starts a separate run. The four classifier notebooks can run independently after 05; missing expanding folds are prepared by the runner.
- **Build dashboard exports:** run **11 → 12**. An existing deployment pin is reused; later comparisons do not silently replace it.
- **Refresh saved observations:** run **13**, using the existing model. Registered future-outcome months are blocked intentionally.

Do not run notebooks concurrently or edit `data/active_run.txt` during execution. A repeated study writes a new timestamped folder; previously inspected periods remain exposed.

Optional commands:

```bash
python prepare.py --experiment all       # Prepare chronological folds
python prepare.py --rerun-all           # Separate reproduction through 10; no deployment
python prepare.py --export-environment  # Record exact installed direct-package versions
python prepare.py --verify-checkpoint   # Read-only Python/model/score/SHAP check
```

## Evaluation and current evidence

The label is a local weekly median price rise of at least **10% in the next exact calendar week**, with coverage/history rules. Missing future outcomes stay unlabelled. Features use current/past data; preprocessing and learned screening use each fold's training data. The training-only threshold sensitivity analysis does not prove 10% is an official or optimal economic threshold.

06–09 each use 16 sequential Optuna trials and three expanding 2025 folds. In 09b, 2024 is added to training while original evaluation rows and fold eligibility stay fixed. Its 24 trials include both inherited controls and search learning rate, number of trees, leaves, minimum child support, L2 regularisation and class weighting. Exact repeated settings reuse their fold metrics. Mean fold average precision (AP) selects parameters before January–March validation is loaded. AP is not ordinary classification accuracy.

Saved local results reviewed on 29 October 2026:

| Longer-history candidate | Mean fold AP | Jan–Mar AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Inherited reference | 0.4148 | 0.3300 | 31.0% | 43.2% | 0.3611 |
| Inherited tuned setting | 0.4247 | 0.3189 | 28.4% | 51.7% | 0.3669 |
| Direct-history Optuna | **0.4370** | 0.3178 | 28.1% | 52.4% | 0.3659 |

The new search selected trial 15: learning rate 0.06485, 400 trees, 31 leaves, minimum child samples 20, L2 penalty 8.4398 and no class weighting. Its selected fold AP rose, but later validation AP/F1 did not improve. At each model's own validation-selected cutoff it found 28 extra surges with 156 extra false alerts versus the inherited tuned setting. Further search is not automatically justified by this result.

January–March selects each cutoff, so its classification metrics are **apparent cutoff-selection results**, not independent test estimates. The new candidate follows 10's fold rule; this does not change the existing dashboard pin. The other classifiers have not received the extra history. Old April–June test AP 0.336/F1 0.414 belong to a different, earlier model. Food-group baselines use training-only rates and matched histories; gains over them are not a price-history-only ablation.

## Data periods and report scope

| Purpose | Source/cutoff period |
|---|---|
| Original classifiers in 06–09 | Final training: January–December 2025 |
| Longer-history classifiers in 09b | Final training: January 2024–December 2025 |
| Expanding-fold evaluation | Later 2025 outcomes after each fold's training cutoff |
| Validation and score-cutoff selection | January–March 2026 |
| Previously exposed test data | April–June 2026; older model results do not transfer to current candidates |
| Dashboard refresh | Later observations, including the saved 19–25 October 2026 snapshot; scoring without retraining |

These are source/cutoff periods; complete-week, coverage and target-horizon rules determine usable rows. The IR's January 2025–June 2026-only wording needs amendment to disclose supplementary 2024 training and separate retrospective refreshes. This extends the data boundary while preserving the local next-week classification/prototype objectives. Do not describe January 2024–October 2026 as the training period of every model.

## Saved data and supporting research

Keep the complete active run, its environment file, shared `data/12_longer_history/` cache and `data/11_historical_audit/` when copying the project. Paths recorded with `run::`/`repo::` are portable; legacy paths can be resolved without changing frozen hashes. Missing or changed inputs need their matching originals, not replacement hashes.

`research/09` records source-quality sensitivity; `research/10` tests matched-shop features; `research/12` preserves the earlier history experiment. Their older checkpoints may require the corresponding earlier Git revision (for example `7cbd4ec`). They are supporting evidence, not extra mandatory training stages.

**Optional new feature experiment:** after a completed 09b direct-history search, open `research/15_relative_price_features.ipynb` with the same verified project kernel and **Restart Kernel and Run All**. It adds seven current/past relative-price features to the frozen longer-history LightGBM settings, compares three expanding folds and then January–March validation, and saves a separate study under the active run's `research/15_relative_price_features/`. It uses existing verified caches, makes no downloads and does not change Notebook 10 or the dashboard. Do not rerun preparation to start it. A 0.70 AP reference is aspirational; no gain or independent accuracy is promised. Share its `fold_summary.csv`, `validation_comparison.csv` and `validation_group_metrics.csv` for interpretation. Numerical feature checks and an isolated synthetic end-to-end execution passed; these do not establish improvement on your real data.

**Cross-district research:** `research/16_cross_district_signals.ipynb` tests six additional signals from other districts for the same item/unit/state/shop type, delayed by one completed week. It excludes the target district, needs at least two peers for price/change summaries, and builds peer observations without next-week outcome eligibility. Run it with the same verified kernel after 09b; Notebook 15 is not a prerequisite. Settings, target rows and labels stay fixed; there is no new parameter search or automatic dashboard replacement. Its own timestamped folder contains fold/validation/group comparisons and `peer_feature_support.csv`. Report AP changes and peer coverage together, including declines. Numerical exclusion/timing checks, synthetic fitting and fresh-kernel execution are technical checks, not evidence of real-data improvement.

**Larger tuning budget:** `research/17_larger_optuna_budget.ipynb` continues from the recorded 24-trial 09b results using the original 22 features and unchanged search ranges. Default `TARGET_TRIALS = 200` means **200 total**, including the inherited 24, with fold-selected comparisons at 24/100/200. Use the verified project kernel and Restart Kernel and Run All; do not rerun preparation. It checkpoints completed trials and can resume, or extend a 100-total run to 200 by changing only that setting. The seeded continuation conditions on recorded trials; the original unsaved sampler state cannot be reproduced exactly. January–March validation is read only after the requested search finishes; the largest-budget fold winner stays the research candidate regardless of later validation ranking. The study saves under the active run's `research/17_larger_optuna_budget/`, with immutable timestamped result folders. Share `budget_comparison.csv`, `fold_summary.csv`, `trial_summary.csv` and `validation_group_metrics.csv`. No automatic dashboard deployment or guaranteed AP gain follows from a larger search.

The latest saved 17 execution records mean fold AP **0.4370 → 0.4382 → 0.4389** and January–March AP **0.3178 → 0.3247 → 0.3327** at 24/100/200 total trials. This is a modest recorded improvement, not AP 0.70, an independent test, or an automatic deployment decision. These saved real-data outputs were inspected; cloud synthetic checks do not reproduce their accuracy.

**Label and anomaly audit:** after completed 05/11, run `research/18_label_and_anomaly_audit.ipynb` with the verified kernel. It reconstructs the 2×2 agreement from 05's saved training-pair diagnostic and compares the deployed pooled detector with a training-only food-group median/IQR-normalised representation, keeping the same six anomaly features and Isolation Forest settings. No per-group flag quotas are imposed. Share `label_sensitivity.csv`, `anomaly_group_comparison.csv` and `anomaly_flag_overlap.csv`; inspect `training_review_candidates.csv` separately for qualitative source-evidence review. A redistribution of flags is not verified accuracy, and alternative labels are not confirmed corrections. The audit loads no row-level surge labels or test rows and changes no target or dashboard. It saves a separate timestamp under the active run's `research/18_label_and_anomaly_audit/`; do not rerun frozen preparation to use it.

**Other classifiers with matching longer history:** the independent notebooks below use the same verified 2024–2025 training populations and evaluation rows as 09b/17. Run them **one at a time** with the verified project kernel; their full-data searches may be expensive. Only the original respective classifier and completed 09b preparation/cache are prerequisites—do not rerun 01–18. Each refits the original 16 parameter configurations on longer history, then extends the search. Old 2025 scores are not reused as longer-history scores. Checkpoints resume compatible progress and timestamped result folders preserve completed evidence.

| Research notebook | Search policy |
|---|---|
| `19_longer_history_logistic_regression.ipynb` | Original C/class-weight range; 100/200 total proposals, including the 16 refitted configurations |
| `20_longer_history_random_forest.ipynb` | Complete all **48 distinct** original depth/leaf/feature/weight configurations |
| `21_longer_history_xgboost.ipynb` | Complete all **54 distinct** original depth/child-weight/L2/weighting configurations |

For RF/XGBoost, 200 proposals cannot add unique settings after those grids are exhausted; recorded trials can exceed the distinct total if the inherited 16 contain duplicates. LR's finite trial budget does not prove a globally optimal C or trial count. Compare original 2025 selected settings against the same settings with history to isolate the history effect, then compare first-16 longer-history selection against the larger search. All selection uses fold AP before later validation is read. Share each notebook's `budget_comparison.csv`, `fold_summary.csv`, `trial_summary.csv` and `validation_group_metrics.csv`. These are new research studies, not automatic additions to Notebook 10 or dashboard deployment. Equal data does not make search spaces equally exhaustive or guarantee AP gains. The original four-model results remain the 2025 studies until the new local runs supply their own evidence.

Notebook 11's detector remains an active dashboard component. The alternative in 18 is currently **research evidence**, not a deployed replacement. It could be integrated later only after defining the desired anomaly meaning, checking cases with credible evidence and reviewing the serving/export changes; more balanced flag counts alone are insufficient.

**Label and anomaly audit:** after completed 05/11, run `research/18_label_and_anomaly_audit.ipynb` with the verified kernel. It reconstructs the 2×2 label agreement from 05's saved training-pair diagnostic and compares the deployed pooled Isolation Forest with one using training-only food-group median/IQR normalisation. Both use the same six anomaly features and detector settings; there are no per-group flag quotas. Compare group flag rates/overlap and review up to 40 prioritised training entries. Redistribution is not verified anomaly accuracy, and a changed label under another weighting policy is not a confirmed error. The audit reads no surge labels or test rows, changes no target/deployment, and saves separate outputs under the active run's `research/18_label_and_anomaly_audit/`. Share `label_sensitivity.csv`, `anomaly_group_comparison.csv` and `anomaly_flag_overlap.csv`. Do not rerun frozen preparation to use it.

For current source review, run **only Section 2** in `research/14_review_readiness.ipynb`. It preserves the log and does not correct prices. Keep the 12 suspicious cases unresolved unless independent evidence supports a decision. Its other workload sections refer to an older checkpoint.

`research/13` registers outcomes for **2–29 November 2026**, with evaluation no earlier than **30 November**, subject to availability. The registration covers its recorded checkpoints, not a substituted candidate, and does not itself prove timely forecasts were archived. Preserve its guard and original rules. Any new independent claim needs an untouched period and rules/checkpoints frozen before inspecting outcomes.

False alerts, missed surges, vegetable-dominated positives, sparse coverage, changing shops and provisional historical metadata remain limitations. Fuel, weather and other external drivers are outside the inputs. SHAP explains model contributions, not economic causes. Usability, stakeholder adoption, savings, affordability and inequality effects have not been established.

## Environment and troubleshooting

The existing checkpoint passed Python 3.12.2 verification on your Mac: 50 saved scores/alerts, cached SHAP consistency and five fresh explanations. This verifies technical compatibility, not future accuracy or source correctness.

- **Existing environment problem:** use the exported `environment_requirements.txt` inside the saved run to restore exact versions in a separate Python 3.12 environment. `--export-environment` groups missing packages/version mismatches into one restore command. It does not install them automatically.
- **Fresh installation only:** create a Python 3.12 virtual environment and install `requirements.txt`. JupyterLab is optional; VS Code uses the listed kernel/execution packages. On macOS, LightGBM may need `brew install libomp`. Broad dependency ranges are not a checkpoint environment lock.
- **Missing/changed inputs:** restore the matching snapshot; keep `data/` and original manifests. Do not retrain merely to open the dashboard.
- **Missing dashboard export:** investigate the existing refresh with 13 or complete 11 → 12 for the pinned model. Preserve pointers and archived exports.
- **Download failure:** uncached data needs network access; verified complete caches avoid downloads. Cloud access has returned proxy 403.

## Review before writing the report

The 30 October 2026 review executed 01–13 end to end on synthetic data, including all four classifier searches, direct-history tuning, anomaly detection, SHAP, refresh and the dashboard export loader. Notebook 10 now reconstructs overall/group metrics from saved validation predictions and checks fold summaries before ranking. Source review preserves editable evidence notes while rejecting changed source-case fields; a zero-case review is valid.

The saved **30 October 2026 Notebook 10 execution completed the new audit for all seven candidates**, saved a new comparison, and confirmed the existing dashboard pin remains unchanged. Its displayed metrics/counts agree with the previously supplied CSVs within display rounding. This completes the requested local comparison check; rerun 10 only after a new model study or an intentional re-audit. Run only Section 2 in `research/14` if reviewing the source log. The existing Mac checkpoint verification also passed. Cloud synthetic checks do not independently reproduce real-data scores: local source/model caches are absent and official source access returned proxy 403.

Before submission, align report dates/metric definitions, state the baseline criteria explicitly, keep validation-cutoff results separate from independent tests, and disclose unresolved prices/group coverage and unmeasured user impact. No additional tuning or dashboard model replacement follows from the review.

## Map and playback

Choose an item/unit and shop type, then play or move the weekly timeline. District selection pauses playback; two district histories can be compared. Exact consecutive weeks are required for percentage changes; gaps stay gaps and grey means insufficient comparable data. Historic prices have no invented model alerts. Reset District to **All** for wider comparisons. Sampled medians do not guarantee the cheapest shop or population coverage.

The bundled [DOSM district boundaries](assets/malaysia-districts.geojson) are the unmodified 160-feature [official file](https://github.com/dosm-malaysia/data-open/blob/21a78e98efd4cd9b022a27a1bf67d167076b7591/datasets/geodata/administrative_2_district.geojson), retrieved 28 October 2026, under the [Open Data License](assets/DOSM-OPEN-DATA-LICENSE.md). Matching handles case, punctuation and equivalent federal-territory prefixes. Unmatched districts stay in tables/selectors; no aliases or boundary changes are guessed and no official endorsement is implied.
