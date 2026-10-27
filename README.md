# HargaLens — PriceCatcher FYP

**See local prices clearly.** A research prototype for local weekly food prices, next-week surge review, separate anomaly flags and precomputed SHAP explanations. Scores are uncalibrated; affordability benefits and stakeholder adoption have not been established.

## Open the dashboard

From the FYP folder:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Open Streamlit's printed address. Stop it with Ctrl+C. **Completed notebooks do not need rerunning just to open the dashboard.**

For a first installation, create the environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In VS Code, select **FYP/.venv/bin/python** as the notebook kernel. An activated terminal does not change the notebook kernel. Data/models are local and are not included in Git.

## The files you need

| Files | Role |
|---|---|
| Notebooks 01–05 | Integrate, inspect, clean and aggregate data; create exact-week features and labels |
| Notebook 06 | Compare models chronologically and preserve the original experiment |
| Notebooks 07–08 | Separate anomaly detector, selected-model SHAP and dashboard exports |
| `app.py` | Read-only dashboard |
| `pricecatcher_config.py` | Shared settings and run locations |
| `prepare.py` | Builds missing evaluation folds when 06 needs them; keep it beside 06 |
| Notebook 09 | Optional observation refresh, without training |
| `research/` | Supporting experiments; not a routine execution checklist |
| `assets/hargalens-mark.png` | Dashboard logo and icon |
| `requirements.txt`, `.gitignore`, this README | Installation and repository instructions |

There are **three Python files**. Keep the research notebooks as evidence, including unsuccessful experiments. Keep completed local runs and source snapshots in `data/`. `.venv/` is your working environment; notebook checkpoints and Python caches can be removed when no process is using them and will be recreated.

## Run notebooks only when needed

| Goal | What to run |
|---|---|
| View completed results | Start `app.py`; no notebook rerun |
| Reproduce the original experiment | **01 → 08**, using Restart Kernel and Run All in each |
| Build the longer-history dashboard from scratch | **01 → 06 → research/11 → research/12 → 07 → 08** |
| Refresh dashboard observations | **09 only**, after completing 08 with the longer-history checkpoint |
| Investigate training source prices | Final cell of **03** and optional **research/14**; leave preparation/model outputs intact |

01 starts a separate reproduction and reuses a verified local source when available. 06 prepares missing expanding folds automatically and reuses an already completed model. `data/active_run.txt` remembers the selected run across kernels; do not edit it or run preparation/modelling concurrently.

The optional terminal command `python prepare.py --rerun-all` reproduces **01–06** in a separate run. It does not run 07–09 or research studies. Normal notebook **Run All** remains supported. Repeating an exposed test period is reproduction, not fresh independent validation.

## Use the dashboard

- **Overview:** location/item filters, Spotlight price stories, food-group counts, watchlist and CSV export. A downloadable HTML story can be printed as a PDF.
- **Explore prices:** one local series, unit, observed medians, coverage and cached explanation. Missing weeks remain blank; unsupported forecasts are explicit.
- **About the model:** the displayed model's own validation results, fixed cutoff, group guidance, coverage and limitations.

Spotlight cards show observed weekly prices, units and changes from the previous week, with separate review and unusual-pattern labels. The selected story pairs its price history with observation support; **Why this score?** opens the explanation summary. The watchlist starts with compact columns; **Show more columns** reveals the full study details, while CSV exports always retain them. **Snapshot dates and coverage** and **Food-group overview** keep supporting information available without crowding the main view. Light and dark themes are supported.

The optional **Priority shortlist** shows up to 25, 50 or 100 above-cutoff alerts within your filters, ranked by score with case ID breaking ties. It shows how many eligible alerts are omitted; **Review alerts** restores the full list. This changes review workload, not model scores, cutoff or main counters. Smaller lists can miss more surges, and their accuracy is not established for new snapshots/filters. Some groups may receive no shortlist places.

Dates, observation age, model identity and study guidance travel with downloads. Group guidance refers to the saved January–March 2026 longer-history study, not confidence in an individual case. Price history remains available for weak-evidence groups.

The app reads verified cached tables through `08_dashboard/dashboard_manifest.json`, or `refresh_manifest.json` when published by 09. It does not load models, calculate SHAP or download prices while browsing. Historical charts read only identifier, price and coverage columns from frozen original inputs, plus refreshed eligible history; they exclude the added 2024 training history. Local explanations cover up to 200 selected latest cases, and the global summary uses a common original-2025 reference sample.

## Optional refresh and source review

In 09, leave `SNAPSHOT_WEEK = None` for the latest fully closed calendar week in Malaysia, or choose an earlier Monday. Source observations must extend through its Sunday. The notebook reuses frozen models, cutoff, eligibility, price bounds, lookups and unit conversions; source/cache checks protect completed runs. Unknown metadata is excluded and counted; statistical flags remain. Scores are retrospective offline outputs, not archived timely forecasts or a new accuracy evaluation.

09 compares support with the original snapshot by group/state. The saved local run supports **2,946 rows for 12–18 October 2026**, versus **5,343** in June. It excludes four September records with unknown shop code 22280 and retains 205 statistical flags across selected July–October source records. Support differences require investigation; they do not establish changed accuracy. These flags are separate from the 12 training-source cases.

Restart/reload the app after refreshing. Failed checks preserve the published pointer. Registered outcome months, including November under the current plan, remain protected. To intentionally reopen the original June export, remove **only** `08_dashboard/refresh_manifest.json`, keep its archived caches and restart the app.

The 12 training-source cases remain unresolved in the saved research evidence. Record independent receipt/listing/provider-correction evidence in the existing `14_review_readiness/source_review/<snapshot>/source_review_log.csv`: `decision`, `evidence_reference`, `evidence_notes`, `reviewer`, `reviewed_on`. Keep a case `unresolved` without sufficient evidence. Neighbour agreement or re-downloading the same price does not confirm correctness. Research 14 validates/preserves user-recorded notes; corrections are not applied automatically. Evaluate confirmed corrections in a separate documented experiment.

## Research evidence and report boundaries

| Study in `research/` | Purpose/finding |
|---|---|
| 09 — source quality | Excluding 58 suspected training records gave small mixed validation gains; not adopted automatically |
| 10 — matched-shop features | Mean fold AP declined slightly; original features retained |
| 11–12 — longer history | Audit 2024, then compare the frozen candidate with additional history; required to reproduce the current longer-history dashboard |
| 13 — future registration | Frozen-model plan for outcomes 2–29 November 2026; earliest evaluation 30 November, subject to data availability |
| 14 — review readiness | Source-review log and exploratory validation-only weekly workload comparisons; no policy adopted |

| Evidence | Result | Comparison |
|---|---|---|
| Original April–June test | AP **0.336**, F1 **0.414** | Current-change baseline: AP 0.151, F1 0.225 |
| Longer-history January–March validation | AP **0.330**, F1 **0.354** | Original validation: AP 0.223, F1 0.332 |
| Longer-history validation alerts | Precision **23.8%**, recall **69.1%** | 2,702 correct alerts, 8,659 false alerts, 1,210 missed surges |

**Original test scores do not establish the longer-history candidate's test performance.** Its independent test performance remains unconfirmed. Vegetables dominate the positive evidence; chicken detection is limited, eggs/flour deteriorated, rice/oil have scarce validation positives, and sugar has no positive training/validation examples. Statistical flags and Isolation Forest outputs are not confirmed errors; no labelled anomaly accuracy is established. SHAP is associative, not causal.

Research 14's validation-only caps of 25/50/100 per week gave precision **56.0%/53.3%/49.5%**, but recall only **4.3%/8.2%/15.2%**; all selected cases were vegetables. These are exploratory whole-period results, not measured performance for dashboard filters or refreshed dates. July–September exposure is uncertain; registration does not establish live forecast availability.

The fixed target is an observed local median increase of **at least 10% in the next exact calendar week**, subject to coverage/history requirements. Missing outcomes stay unlabelled. Screening, bounds and imputation use the appropriate training period; lags and rolling features are past-only. Original expanding folds are within 2025; the holdout trains through December 2025, selects its score cutoff on January–March 2026 and evaluates on April–June 2026. The percentage-point label tolerance is `1e-10`; legacy schema-2 runs retain their strict comparison.

**Threshold evidence:** Notebook 05, section 6, already compares 5%, 10%, 15% and a training-percentile threshold on training observations only. Its `training_threshold_comparison.csv` records overall and food-group counts/rates for supported pairs and model-ready rows. Use this as sensitivity evidence for the fixed 10% working definition, not proof of an economically optimal threshold. Reading the existing saved table does not require rerunning preparation or models.

Sparse coverage, changing shop composition, provisional historical lookup assumptions, unresolved prices and absent external economic drivers limit conclusions. No universal model superiority, affordability improvement, reduction in inequality or stakeholder adoption has been demonstrated. Preserve source snapshots, package versions and saved findings when writing the report.

## Optional model-specific tuning

`research/15_logistic_regression_optuna.ipynb` is the first separate model study: Logistic Regression with a 16-trial Optuna search over regularisation strength (`C`) and class weighting (`None` or `"balanced"`). After Notebook 06 has prepared the three folds, Restart Kernel and Run All. Install the updated requirements in the selected kernel first. Outputs go to a separate `15_logistic_regression_optuna/<timestamp>/` folder inside the active data run. Existing models and dashboard inputs are not replaced.

The study compares the original two balanced Logistic Regression settings, their unweighted counterparts and simple baselines on identical 2025 folds, then reports a separate exploratory Jan–Mar 2026 validation comparison. It reads no test data. Reused periods and cutoff-selection results are not independent evidence of improvement. Keep the previous 12-trial C-only outputs as earlier exploratory evidence; the expanded search writes a new timestamped folder. In summary CSVs, `none` denotes unweighted training. This study is optional; it is not a new step in the dashboard workflow.

`research/16_random_forest_optuna.ipynb` is Model 2: a standalone Random Forest study using the same folds, AP objective and 16-trial budget. It tunes depth, leaf support, feature sampling and class weighting, with 150 trees fixed. Run All using existing prepared inputs; completing Logistic Regression first is optional. Outputs go to `16_random_forest_optuna/<timestamp>/` within the active data run. Compare only matching input periods/rows; neither study reads test data or replaces the dashboard model.

`research/17_xgboost_optuna.ipynb` is Model 3: XGBoost with the same chronological protocol and 16-trial budget. It tunes depth, minimum child weight, L2 regularisation and training-only positive weighting, with 150 rounds and learning rate 0.05 fixed. Reuse the prepared inputs and Run All; other model studies do not need rerunning. Separate outputs go to `17_xgboost_optuna/<timestamp>/` within the active data run. Existing results and dashboard checkpoints remain unchanged.

`research/18_lightgbm_optuna.ipynb` is Model 4: LightGBM with the same chronological protocol and 16-trial budget. It tunes leaf count, minimum child samples, L2 regularisation and class weighting, with 150 rounds and learning rate 0.05 fixed. Run All using existing prepared inputs. Outputs go to `18_lightgbm_optuna/<timestamp>/` within the active data run. This 2025-trained exploratory study does not replace the longer-history dashboard model or access test data. Compare matching periods and rows across studies 15–18.

Each model study (15–18) contains its own implementation explanation, tuning outputs, validation results and **section 7 report notes**. The notes interpret the saved execution reviewed on 27 October 2026; update them if a future run changes the results. They distinguish fold selection, apparent cutoff-selection performance and food-group limitations. Reading the notes needs no rerun. Notebook 06 remains the original reference experiment and research/12 remains the separate longer-history/dashboard study.

## Troubleshooting

- **Missing dashboard files:** run 08 after restoring/completing its matching checkpoints. For a published refresh, verify with 09. Do not restart 01 merely to reopen results.
- **Missing package:** select the project kernel, run `%pip install -r requirements.txt` in a temporary main-notebook cell, then restart. For macOS LightGBM's `libomp.dylib` error, use `brew install libomp` in Terminal.
- **Changed frozen files:** restore the matching snapshot and package versions. Do not replace hashes or delete `data/`. Start 01 only for a deliberate separate reproduction.
- **Download failure:** new source access requires the official endpoint. A complete verified cache avoids downloads; this cloud has returned proxy 403. Partial downloads are not accepted as complete data.
