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

## Save and restore the project

Git excludes real data, fitted models and dashboard exports. A fresh clone cannot display your actual results without the matching local files. Preserve:

- The complete active `data/runs/run_.../` folder and `data/active_run.txt`.
- Shared `data/12_longer_history/` caches and `data/11_historical_audit/`.
- The active run's `environment_requirements.txt`, alongside the Python version used for training.

Keep saved manifests, deployment pointers and archived exports together. `run::`/`repo::` input references are portable; legacy paths are resolved without replacing their hashes. Do not delete saved runs or change hashes to repair a mismatch.

## Supporting research — optional

These notebooks are outside the main run sequence. Unsuccessful experiments remain useful evidence; they do not automatically replace the dashboard model. Use the verified project kernel and run one notebook at a time.

| Notebook in `research/` | Purpose and status |
|---|---|
| `09_source_quality_sensitivity.ipynb` | Historical training-source sensitivity; requires its original checkpoint |
| `10_matched_shop_feature_experiment.ipynb` | Historical matched-shop feature comparison; requires its original checkpoint |
| `12_longer_history_experiment.ipynb` | Earlier history experiment; preserves the evidence behind the longer-history workflow |
| `13_register_future_evaluation.ipynb` | Preserve any existing future registration; this legacy tool is not a new training stage |
| `14_review_readiness.ipynb` | **Section 2 only** for the current source-review log; other sections refer to an older checkpoint |
| `15_relative_price_features.ipynb` | Seven relative-price features; no reliable overall improvement established |
| `16_cross_district_signals.ipynb` | Six lagged peer-district signals; mixed fold/validation results |
| `17_larger_optuna_budget.ipynb` | LightGBM search-budget comparison at 24/100/200 total trials |
| `18_label_and_anomaly_audit.ipynb` | Training-label sensitivity and an alternative anomaly representation |
| `19_wholesale_availability_audit.ipynb` | Source-only wholesale coverage audit; no training or downloads |

Research 09/10/12 use the old combined-model checkpoint, not today's Logistic Regression notebook. Their original saved artifacts and matching Git revision (for example `7cbd4ec`) may be required. Keep them as historical evidence rather than running them against current classifier outputs.

Research 15–17 require completed 09b results and matching caches; 18 requires completed 05/11. They save separate outputs under the active run's `research/` folder. Do not rerun preparation to start them. Parameters are selected using expanding-fold AP; January–March results remain exploratory because that period has already been inspected.

Research 17 defaults to **200 total proposals**, including the inherited 24. Completed trials are checkpointed, so compatible progress resumes; changing 100 to 200 extends the same study. The original sampler state was not saved, so continuation is seeded but cannot reproduce that earlier state exactly. Saved mean fold AP was **0.4370 → 0.4382 → 0.4389**, and January–March AP was **0.3178 → 0.3247 → 0.3327** at 24/100/200. This is a modest recorded gain, not an independent test or evidence that AP 0.70 is attainable.

Research 18 compares label weighting and pooled Isolation Forest representations. Different labels do not establish source errors; redistributed anomaly flags do not establish improved detection accuracy. Notebook 11's pooled detector remains the dashboard component. Review `training_review_candidates.csv` with independent source evidence before drawing case-level conclusions.

The optional longer-history LR/RF/XGBoost search notebooks were removed from this workflow. Existing local research outputs can be retained for report evidence without being rerun or included in dashboard deployment. Removing their notebooks does not remove their local result folders.

For the wholesale audit, open `research/19_wholesale_availability_audit.ipynb` in the verified project kernel and **Restart Kernel and Run All**. It reads only 2024–2025 source rows from existing caches and saves five compact CSVs plus a manifest under the active run's `research/19_wholesale_availability_audit/`. Missing 2024 history is reported, not downloaded. Share `source_summary.csv` and `group_signal_coverage.csv` first. This measures potential lagged-signal coverage, not predictive accuracy; prepared labels, models and dashboard exports stay unchanged.

## Environment and troubleshooting

- **Saved checkpoint:** use its exported `environment_requirements.txt` in a separate matching Python 3.12 environment. `--export-environment` reports grouped missing-package/version restore instructions; it does not install packages.
- **Fresh installation:** install `requirements.txt` in Python 3.12. These broad dependency ranges are not a checkpoint lock. On macOS, LightGBM may require `brew install libomp`.
- **Missing or changed inputs:** restore the matching snapshot and original manifests. Do not retrain merely to open the dashboard.
- **Missing dashboard export:** check the existing refresh in 13 or complete 11 → 12 for the pinned model. Preserve deployment pointers and archived exports.
- **Download failure:** uncached data needs working source access. Verified complete caches avoid repeat downloads; earlier cloud runs encountered proxy 403 responses.

`python prepare.py --verify-checkpoint` checks the pinned classifier, saved scores/alerts and SHAP consistency without fitting or publishing. The Mac checkpoint passed a 50-row score/alert check and five fresh explanations. These checks establish technical compatibility, not future accuracy or source correctness.

## Before writing the report

The main workflow has passed isolated synthetic execution checks. The saved 30 October Notebook 10 execution also completed the comparison audit for all seven candidates while preserving the dashboard pin. These checks do not independently reproduce real-data accuracy in the cloud.

Keep these distinctions clear:

- Compare the four original classifiers on their shared 2025 training data. Present added 2024 history as a separate experiment.
- Name the model and cutoff associated with each result. Research 17's candidate and the pinned dashboard classifier are different checkpoints.
- January–March cutoff-selection metrics are exploratory. Older April–June results do not transfer to newer models.
- Preserve existing November registrations. They cover only the recorded checkpoints and rules. The proposed outcome period is **2–29 November 2026**, with evaluation no earlier than **30 November**, subject to availability. Registration had to precede 1 November; do not create a late registration and describe it as prospective. Timely forecast archiving must also be demonstrated separately.
- False alerts, missed surges, vegetable-dominated positives, sparse coverage, changing shops and provisional historical metadata remain limitations. Keep the 12 suspicious source cases unresolved unless independent evidence supports a decision.
- Fuel, weather and other external drivers are outside the inputs. SHAP shows model contributions, not economic causes. Stakeholder adoption, savings, affordability and inequality effects have not been demonstrated.

## Map and playback

Choose an item/unit and shop type, then play or move the weekly timeline. District selection pauses playback; two district histories can be compared. Exact consecutive weeks are required for percentage changes; gaps stay gaps and grey means insufficient comparable data. Historic prices have no invented model alerts. Reset District to **All** for wider comparisons. Sampled medians do not guarantee the cheapest shop or population coverage.

The bundled [DOSM district boundaries](assets/malaysia-districts.geojson) are the unmodified 160-feature [official file](https://github.com/dosm-malaysia/data-open/blob/21a78e98efd4cd9b022a27a1bf67d167076b7591/datasets/geodata/administrative_2_district.geojson), retrieved 28 October 2026, under the [Open Data License](assets/DOSM-OPEN-DATA-LICENSE.md). Matching handles case, punctuation and equivalent federal-territory prefixes. Unmatched districts stay in tables/selectors; no aliases or boundary changes are guessed and no official endorsement is implied.
