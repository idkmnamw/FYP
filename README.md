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
