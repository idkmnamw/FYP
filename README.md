# PriceCatcher FYP

A readable pipeline for Malaysian local weekly prices, next-week surge prediction, anomaly review and precomputed explanations. The dashboard app is the next stage; it is not implemented yet.

## Start here

**Already completed the longer-history experiment?** Pull the latest code and Run All in **08 only** using your project kernel. It exports fresh scores and SHAP for that checkpoint, preserving the original results. No rerun of 01–07 or the research studies is needed.

**Starting a new reproduction?** Run **01 → 06**, then **07 → 08**. Notebook 01 starts a separate run; the others continue it automatically. For a longer-history candidate, run research **11 → 12** once after 06, then rerun 08. Studies 09, 10 and 13 are optional evidence, not prerequisites.

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
pricecatcher_config.py              shared settings and run locations
prepare.py                          preparation runner used internally by 06
requirements.txt
research/                           optional studies 09–13
```

Only two Python files support the notebooks. Generated data, models and caches stay under ignored `data/`; none need committing to Git. Keep completed experiment folders for the report.

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

**Important:** the original model's April–June 2026 test scores are not test scores for the longer-history candidate. Its January–March 2026 improvement remains exploratory validation evidence; independent test performance is unconfirmed. Saved 08 displays were cleared because the handoff changed—Run All locally for current outputs.

## Research evidence: optional, not the everyday workflow

| Notebook in `research/` | Purpose and conclusion from reviewed real outputs |
|---|---|
| 09 — source quality | Excluding 58 suspected training records gave small mixed validation gains; not adopted automatically |
| 10 — matched-shop features | Mean fold AP declined slightly; original feature list retained |
| 11 — historical audit | All 12 monthly 2024 files available/compatible; January, July and December inspected |
| 12 — longer history | Validation AP improved about 0.223 → 0.330; eggs/flour deteriorated, so report trade-offs |
| 13 — future registration | Optional frozen-model November plan; no prices, predictions or evaluation loaded |

The studies retain their saved outputs and generated experiment folders. Moving notebooks changes code provenance; deliberately rerunning a study may create a new fingerprint folder. It does not replace the completed evidence. Do not rerun the studies just to refresh the dashboard.

Study 11 saves small audit reports under `data/11_historical_audit/`. Study 12 reuses the audited 2024 source cache, retains unknown metadata in that cache, and excludes only records lacking required modelling identifiers from supplementary training. `historical_metadata_review.csv` records the affected codes/fields; `historical_cleaning.csv` reports attrition. Current lookup units/geography remain a provisional historical interpretation.

To inspect suspicious prices in a completed run, run **only the final cell of 03**. Its `training_source_context.csv` is regenerated evidence, not an editable decision log. Priorities are investigation aids; they do not confirm errors or replacement prices. Keep research decisions and independent evidence in report notes.

Study 13 reserves four outcome weeks, 2–29 November 2026, with earliest evaluation 30 November subject to full data availability. It is optional and should not delay dashboard work or the December report. July–September exposure is uncertain, so do not claim it is an untouched test. Registration alone does not implement evaluation or prove live forecasts.

## Method and interpretation

A surge is an observed local median increase of at least **10% in the next exact calendar week**. This research target is not an official standard or a fixed-shop price index. A `1e-10` percentage-point tolerance handles floating-point arithmetic; legacy schema-2 runs retain their original strict comparison. Missing outcomes stay unlabelled. Feature imputation is fitted only on each training fold.

Original folds expand within 2025. Holdout trains through December 2025, selects its cutoff on January–March 2026 and evaluates on April–June 2026. Those outcomes have been inspected: reproductions and later development on them are not fresh independent validation. The longer-history candidate uses the original target/features/cutoff and identical validation rows, with training history extended back to 2024.

Report false alerts, missed surges, scarce group positives, unsupported places, source uncertainty, aggregation sensitivity and time variation. Suspicious positive prices are not automatically errors. The six original retrospective exclusions remain a limitation. Supplementary paired week-bootstrap intervals are descriptive over a short period, not proof across seasons. No anomaly ground truth, stakeholder adoption or affordability improvement has been established.

## Optional terminal runner

```bash
python prepare.py --rerun-all
```

This creates a separate reproduction and runs **01–06** with fresh kernels; 07, research studies and 08 remain interactive. Manual Run All is supported. `prepare.py` stays because 06 calls it for missing folds.

## Next stage

Build one Streamlit `app.py` using the 08 manifest and cached tables; weekly trends/coverage will also need validated prepared history. Do not train models or calculate SHAP while users browse. Evaluate correctness, loading time, filter behaviour and whether users understand alerts, dates and insufficient coverage. No further model experiment is required to finish the prototype.
