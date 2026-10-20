# HargaLens — PriceCatcher FYP

**HargaLens — See local prices clearly.** A readable pipeline for Malaysian local weekly prices, next-week surge prediction, anomaly review and precomputed explanations. A single-file Streamlit dashboard displays the saved forecasts, historical prices and explanations without training models.

## Start here

**Already completed Notebook 08?** Pull the latest code, install the updated requirements, and start the dashboard using the command below. No notebook rerun is needed for the completed longer-history export. If `08_dashboard/dashboard_manifest.json` is missing, Run All in **08 only** using your project kernel first.

**Starting a new reproduction?** Run **01 → 06**, then **07 → 08**. Notebook 01 starts a separate run; the others continue it automatically. For a longer-history candidate, run research **11 → 12** once after 06, then rerun 08. Studies 09, 10, 13 and 14 are optional evidence, not prerequisites.

Do not start 01 merely to reopen saved results. Do not run preparation and modelling concurrently. The printed run folder identifies the files used.

## Repository layout

```text
01_integrate_pricecatcher.ipynb
02_data_understanding_and_quality.ipynb
03_cleaning_and_outliers.ipynb
04_weekly_aggregation.ipynb
05_features_and_labels.ipynb
06_model_training_and_evaluation.ipynb
07_anomaly_detection.ipynb
08_shap_and_dashboard_inputs.ipynb
app.py                              read-only Streamlit dashboard
pricecatcher_config.py              shared settings and run locations
prepare.py                          preparation runner used internally by 06
requirements.txt
research/                           optional studies 09–14
assets/hargalens-mark.png            shared dashboard logo and page icon
```

Two Python files support the notebooks; the third, `app.py`, is the dashboard. Generated data, models and caches stay under ignored `data/`; none need committing to Git. Keep completed experiment folders for the report.

Keep the eight main notebooks, the three Python files, the logo under `assets/`, `requirements.txt`, `.gitignore` and this README. The six `research/` notebooks preserve the comparisons and audit behind the report; they are optional to run. There are no duplicate source files to delete. `pricecatcher_config.py` holds shared rules, and 06 uses `prepare.py` to prepare missing validation folds.

For local housekeeping, `.ipynb_checkpoints/` and `__pycache__/` can be removed when notebooks are stopped; Python/Jupyter recreate them. Keep `.venv/` for your working kernel. Keep completed `data/` runs and source snapshots until they are backed up and no longer needed for the report; deleting them can break model verification or require downloading the data again.

## Install and choose the kernel

From the FYP folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In VS Code, select **FYP/.venv/bin/python** as the notebook kernel. Activating a terminal environment does not change a notebook's kernel. If a package is missing, run `%pip install -r requirements.txt` in a temporary main-notebook cell, restart that kernel, and Run All. Research notebooks can be opened from either the project folder or `research/`.

For JupyterLab, use `python -m jupyter lab`. Restore an older saved model using the package versions in its manifest. On macOS, a LightGBM `libomp.dylib` error needs `brew install libomp` in Terminal.

A complete cached source avoids downloads. Without it, 01 needs `storage.data.gov.my`; this cloud's proxy blocks that endpoint, while the Mac has demonstrated access. No download failure is treated as a complete dataset.

## Main workflow

| Notebook | What to read or run |
|---|---|
| 01 | Start a new run; integrate sources, check joins and standardise egg units |
| 02 | Learn item coverage eligibility from the relevant training period |
| 03 | Clean invalid records, retain statistical flags and export review evidence |
| 04 | Aggregate complete Monday–Sunday weeks and record coverage |
| 05 | Build exact-week history, the fixed target, chronological splits and coverage audits |
| 06 | Preserve the original model comparison/test experiment; supporting diagnostics follow the main results |
| 07 | Fit/reuse a separate training-only Isolation Forest with a 2025 reference |
| 08 | Verify checkpoints, explain the chosen forecast model and publish dashboard inputs |

Notebook 06's helper definitions are divided into labelled sections. Its calculations, selection rules, original cutoff and frozen results are unchanged by the repository cleanup.

### Reopening results safely

- Run All in **06** to reopen the original reference experiment without refitting a completed model.
- Run All in **07** to reuse its detector and regenerate anomaly tables.
- Run All in **08** to reopen/regenerate the derived dashboard cache without fitting either model.
- To repeat preparation, start at **01**. Completed inputs remain protected from accidental overwrite.

`data/active_run.txt` remembers the run across kernels. Do not edit it during execution. Explicit `PRICECATCHER_DATA_DIR` and `PRICECATCHER_EXPERIMENT` overrides remain supported for advanced use. Remove old manually added environment/path cells when using the automatic workflow.

## Dashboard model and exports

**Chosen prototype candidate: longer 2024–2025 history**, when a verified completed study 12 exists. Selection is the latest compatible completed run by completion time, not by score ranking. Notebook 08 prints the exact checkpoint/version and verifies all recorded provenance. If no longer-history run exists, a new dashboard export clearly uses the original model. Once a longer-history dashboard has been published, a missing checkpoint cannot silently downgrade it.

The separate anomaly detector remains trained on 2025. A model change does not turn anomaly flags into surge predictions or require retraining that detector.

The stable dashboard entry point is:

```text
<printed run>/08_dashboard/dashboard_manifest.json
```

It identifies the forecast model/version, training dates, cutoff, **validation evidence/status**, separate anomaly reference and model-specific cache. Three tables in that cache provide:

| File | Use |
|---|---|
| `latest_dashboard.parquet` | All supported latest rows, chosen-model scores/alerts, separate anomaly flags and explanation availability |
| `shap_explanations.parquet` | Up to 200 latest highest-score explanations plus a fixed historical reference sample |
| `reference_feature_importance.csv` | Descriptive SHAP summary on 200 randomly sampled **original 2025 training inputs** |

The historical reference is not a representative sample of the longer model's full training population. Tree explanations use the chosen model's training-path reference; linear explanations use the explicitly declared 2025 background. Contributions reconstruct the chosen model output. Scores are uncalibrated, SHAP is not causal, and local file-read timing is not dashboard latency.

The old `08_explanations/` original-model cache is preserved. New exports use `08_dashboard/<model-hash>/`; no old model, original prediction table or test result is overwritten. The cache manifest verifies provenance/hashes; the stable entry point is published only after output checks pass.

**Important:** the original model's April–June 2026 test scores are not test scores for the longer-history candidate. Its January–March 2026 improvement remains exploratory validation evidence; independent test performance is unconfirmed. The latest saved 08 execution verifies the longer-history export, including additive explanations and 200 cached latest cases.

## Research evidence: optional, not the everyday workflow

| Notebook in `research/` | Purpose and conclusion from reviewed real outputs |
|---|---|
| 09 — source quality | Excluding 58 suspected training records gave small mixed validation gains; not adopted automatically |
| 10 — matched-shop features | Mean fold AP declined slightly; original feature list retained |
| 11 — historical audit | All 12 monthly 2024 files available/compatible; January, July and December inspected |
| 12 — longer history | Validation AP improved about 0.223 → 0.330; eggs/flour deteriorated, so report trade-offs |
| 13 — future registration | Optional frozen-model November plan; no prices, predictions or evaluation loaded |
| 14 — review readiness | Training-source investigation log and validation-only weekly alert workload comparisons; no refit or policy adoption |

The studies retain their saved outputs and generated experiment folders. Moving notebooks changes code provenance; deliberately rerunning a study may create a new fingerprint folder. It does not replace the completed evidence. Do not rerun the studies just to refresh the dashboard.

Study 11 saves small audit reports under `data/11_historical_audit/`. Study 12 reuses the audited 2024 source cache, retains unknown metadata in that cache, and excludes only records lacking required modelling identifiers from supplementary training. `historical_metadata_review.csv` records the affected codes/fields; `historical_cleaning.csv` reports attrition. Current lookup units/geography remain a provisional historical interpretation.

To inspect suspicious prices in a completed run, run **only the final cell of 03**. Its `training_source_context.csv` is regenerated evidence, not an editable decision log. Priorities are investigation aids; they do not confirm errors or replacement prices. Keep research decisions and independent evidence in report notes.

Study 13 reserves four outcome weeks, 2–29 November 2026, with earliest evaluation 30 November subject to full data availability. It is optional and should not delay dashboard work or the December report. July–September exposure is uncertain, so do not claim it is an untouched test. Registration alone does not implement evaluation or prove live forecasts.

### The two remaining practical checks

Run **research/14_review_readiness.ipynb** after the completed longer-history export in 08. No rerun of 01–13, model fitting or dashboard changes is needed. If the source-context CSV is missing, run **only the final cell of 03** first.

The notebook verifies the chosen checkpoint and frozen validation population, prepares 12 high-priority training-price cases and checks their prices/units against the verified integrated snapshot. It preserves a manual evidence log at `<active run>/14_review_readiness/source_review/<snapshot>/source_review_log.csv`. Record independently supported findings there using `unresolved`, `confirmed_error` or `confirmed_valid`, with evidence reference, notes, reviewer and review date. Neighbour agreement and re-downloading the same record do not independently confirm an error. Log confirmations are user-recorded claims, not automatically authenticated. No log entry changes prices, labels, exclusions or models.

The other section compares the **frozen alert rule** against reviewing at most **25, 50 or 100 existing alerts per week**, across the same whole-study validation population. It reports cases/week, precision, recall, F1, false alerts, missed surges and group support. Scores below cutoff are never promoted. Ties use deterministic series-key order. These are illustrative workloads, not stakeholder requirements. Smaller lists can lose recall, precision gains are not guaranteed, and AP is unchanged. Do not apply these metrics to arbitrary dashboard filters or describe this as independent test performance.

Separate fingerprint folders contain `policy_comparison.csv`, `weekly_workload.csv`, `policy_by_group.csv` and `review_manifest.json`. Reruns preserve evidence notes and never load test rows, replace the dashboard or change November registration. The source provider is still inaccessible from this cloud (proxy 403), so independent confirmation remains outstanding. Run the notebook locally for the real-data workload figures; summary CSV metrics alone cannot reconstruct ranked-shortlist results.

Ask intended reviewers what weekly case volume they can investigate, then assess task completion, dates/units, alert versus anomaly comprehension and whether missing coverage/explanations are understood. Select any future policy based on a declared user need and evaluate it prospectively; this notebook intentionally selects no winner. Confirmed material source corrections belong in a separate documented experiment, preserving the current evidence and registered comparison.

## Current saved results

These are the saved real-data displays committed with the notebooks, not a new retraining. The 2025 reference results below are the latest saved reproduction of an already exposed test period.

| Evidence | Reference model | Comparison |
|---|---:|---:|
| 06: April–June test average precision | 0.336 | Current-change baseline: 0.151 |
| 06: April–June test F1 | 0.414 | Current-change baseline: 0.225 |
| 12: January–March validation average precision | Original: 0.223 | Longer history: 0.330 |
| 12: January–March validation F1 | Original: 0.332 | Longer history: 0.354 |

The original test has **3,985 correct alerts, 8,131 false alerts and 3,140 missed surges**. Its precision is 32.9% and recall 55.9%. It beats the supplementary food-group prevalence baseline too (test AP 0.225; F1 0.367), but still needs human review.

Longer history improves validation recall from 65.7% to 69.1%, with 2,702 correct alerts, 8,659 false alerts and 1,210 missed surges. Precision is only 23.8%: **76.2% of its validation alerts are false**. Vegetables account for 2,686 of its 2,702 correct alerts; eggs and flour deteriorate. Sugar has no positive training/validation examples, and cooking-oil positives remain too scarce for dependable conclusions. These validation results cannot be compared directly with the original model's test scores to rank the models.

Coverage remains a practical limitation: 05 reports 3,271 source series with no usable modelling rows. Its latest supported snapshot has **5,343 rows for the week starting 22 June 2026**. It is historical, not a current October/November forecast. The separate 07 detector flags 45 of those rows; without anomaly ground truth, this is a review shortlist, not measured detection accuracy. Unusual source prices remain unresolved, not verified errors.

The latest real-data 08 outputs are committed: the longer-history checkpoint was selected, SHAP additivity passed and the dashboard entry point was written. The app reads that export and verifies its cache and frozen historical inputs before displaying it. None of these results establishes stakeholder adoption, affordability improvement or reliable coverage across every food group.

## Method and interpretation

A surge is an observed local median increase of at least **10% in the next exact calendar week**. This research target is not an official standard or a fixed-shop price index. A `1e-10` percentage-point tolerance handles floating-point arithmetic; legacy schema-2 runs retain their original strict comparison. Missing outcomes stay unlabelled. Feature imputation is fitted only on each training fold.

Original folds expand within 2025. Holdout trains through December 2025, selects its cutoff on January–March 2026 and evaluates on April–June 2026. Those outcomes have been inspected: reproductions and later development on them are not fresh independent validation. The longer-history candidate uses the original target/features/cutoff and identical validation rows, with training history extended back to 2024.

Report false alerts, missed surges, scarce group positives, unsupported places, source uncertainty, aggregation sensitivity and time variation. Suspicious positive prices are not automatically errors. The six original retrospective exclusions remain a limitation. Supplementary paired week-bootstrap intervals are descriptive over a short period, not proof across seasons. No anomaly ground truth, stakeholder adoption or affordability improvement has been established.

## Optional terminal runner

```bash
python prepare.py --rerun-all
```

This creates a separate reproduction and runs **01–06** with fresh kernels; 07, research studies and 08 remain interactive. Manual Run All is supported. `prepare.py` stays because 06 calls it for missing folds.

## Run the dashboard

From your FYP folder with `.venv` activated:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use the local address printed by Streamlit. Stop the app with Ctrl+C. No API keys, database, model training or live downloads are needed. Generated data stays local; Git does not contain the data required to run the app on another machine.

The app follows the same `data/active_run.txt` as the notebooks (or an explicit `PRICECATCHER_DATA_DIR`). It never searches older completed runs to hide an unfinished active run. Missing/changed files produce an actionable message instead of substituting forecasts. It verifies the 08 entry point, cache hashes, model/version consistency, dates, alerts and explanation mapping. Original train/validation/test parquet hashes must match the 08 provenance before history is loaded; only identifier, price and coverage columns are read, not outcome labels. No model is loaded or fitted. Streamlit caches verified data across filter changes and rechecks when relevant file metadata changes.

HargaLens combines **harga** (price) and **lens** (a clearer view). The original transparent basket-and-lens logo uses teal and amber and is stored once under `assets/`; it is reused for the header, sidebar and browser icon. No extra branding library or service is needed. The tagline describes price visibility, not proven savings. The first-visit guide explains the reading order and the distinction between next-week review alerts and observed unusual patterns. Theme-safe card colours work in light and dark mode.

Three tabs keep the workflow small:

- **Overview:** HargaLens Spotlight with three highest-score supported cases, a selectable price story and downloadable evidence brief; resettable state/district/food/item/shop filters, food-group alert-count chart, frozen-threshold alerts, separate unusual-pattern flags, scores and CSV download. These are investigation priorities, not verified event probabilities.
- **Explore prices:** one local series at a time, 12-week/26-week/full-history chart views, RM and unit, missing weeks left blank, current record/shop/day counts and cached SHAP when available. The default is the highest-score supported series within your filters. Historical locations without a latest prediction explicitly show no supported forecast.
- **About the model:** the selected model's own validation evidence, false-alert rate, training dates, cutoff, filtered support counts and limitations. No original-model test scores are attributed to the longer-history candidate.

History comes from eligible **original 2025–June 2026 modelling rows** plus the latest snapshot, not all raw source observations or the added 2024 training data. Boundary, sparse and excluded weeks can appear as gaps. Areas absent from these inputs cannot be assessed. The app is a historical demonstration; the forecast period shown is 29 June–5 July 2026, not a live warning. No filter changes the frozen model cutoff or the 10% target.

Do not delete `data/` or rerun 01 just to open the app. To switch runs intentionally, complete the new run through 08 first. More explanations or newer price data require a separate, verified offline preparation/export step; the dashboard does not generate either while users browse.

### HargaLens Spotlight

Spotlight connects a saved forecast with its local context. The three highest-scoring supported cases in the current filters each have a **Read this price story** button. The selected story shows the observed price/unit, exact forecast period, separate anomaly status, shop/record/day counts, 12 calendar weeks of eligible historical prices and the strongest cached SHAP contributions. Missing explanations are explicit. Scores and alert cutoffs do not change when a story is selected. If every case is below cutoff, Spotlight says so; the shortlist is not presented as confirmed surges.

**Download brief · [item] · [district]** saves a standalone HTML document, readable without the app. It carries the dates, original checkpoint identity, frozen cutoff, observation support, selected model's evidence status/validation precision, explanation availability and recent price table. Open it in a browser and print to save a PDF. Dynamic text is escaped. No AI text-generation service, new model, future price estimates, live data or extra dependency is involved. This is a usability feature that makes the existing evidence easier to inspect and share; it does not establish scientific novelty or improved forecasting performance.

## Evaluate the dashboard before writing final claims

Check representative items and locations against Notebook 08, including the forecast dates, units, scores, alert cutoff, anomaly flags and whether SHAP is available. Check a historical series without a latest forecast; it must not substitute an older prediction. Measure application/browser loading and filter response separately from the notebook's file-read timing.

Ask suitable users to find a local item, interpret its price chart and alert, distinguish an anomaly flag from a forecast, and recognise a missing explanation or unsupported forecast. Record task completion, response times, misunderstandings and improvements. User comprehension/usefulness testing does not establish reduced prices or affordability benefits. Keep the original model's test results and the longer model's exploratory validation evidence separate in the report. No further model experiment is required to implement the prototype.
