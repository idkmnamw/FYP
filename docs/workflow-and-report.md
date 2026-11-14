# Workflow, artifact paths and report scope

This is a map of the current source code, not a record of executing it. Main notebook numbers and research notebook numbers are independent. Paths below are relative to the active run unless explicitly prefixed with `data/`. Restore matching artifacts before treating any result as verified.

## Main notebook to output map

`data/active_run.txt` selects `data/runs/run_.../`. Source caches shared between runs live directly under `data/`. The notebook filename identifies a workflow stage; the output folder identifies the artifact format established when that stage was written. Different numbers are intentional retained paths, not instructions to move files.

| Main notebook | Principal output location | Dependency / interpretation |
|---|---|---|
| 01 integration | `01_integrated/`, including `lookups/` and `source_manifest.json` | Verified source snapshot; an interactive new reproduction starts a separate run |
| 02 quality | `experiments/<experiment>/02_quality/` | Training-only item screening |
| 03 cleaning | `experiments/<experiment>/03_cleaned/` | Daily prices, cleaning metadata and source-review context |
| 04 weekly aggregation | `experiments/<experiment>/04_weekly/` | Complete-scope weekly prices and preparation settings |
| 05 features/labels | `experiments/<experiment>/05_modelling/` | Splits, feature/label settings, coverage diagnostics and latest inputs |
| 05b historical audit | **Shared:** `data/11_historical_audit/` | Audited 2024 source availability; not a classifier |
| 06 Logistic Regression | `06_logistic_regression/<timestamp>/` | Separate 2025-training classifier study |
| 07 Random Forest | `07_random_forest/<timestamp>/` | Separate 2025-training classifier study |
| 08 XGBoost | `08_xgboost/<timestamp>/` | Separate 2025-training classifier study |
| 09 LightGBM | `09_lightgbm/<timestamp>/` | Separate 2025-training classifier study |
| 09b longer-history LightGBM | `09b_longer_history_lightgbm/<timestamp>/{reference,tuned,optuna}/` | Adds 2024 training; shared selected-source cache under `data/12_longer_history/2024_source/` |
| 10 comparison | `10_model_comparison/<timestamp>/` | Comparison and candidate under the declared fold rule; does not replace an existing pin |
| 11 anomalies/deployment | `11_deployment/` and `11_anomalies/<identity>/` | Creates a classifier pin only if absent; fits/reuses the separate anomaly detector |
| 12 explanations/dashboard | `08_dashboard/deployments/<deployment-id>/<export-id>/` | Explanation cache and dashboard manifests; `08_dashboard/active_deployment.json` identifies the publication |
| 13 optional refresh | `09_refresh/<week>/<source-id>/` and refresh exports in the selected dashboard directory | Derived offline observations/scores with the original model and cutoff |
| `app.py` | No training or dataset outputs | Reads saved dashboard tables and prepared price history |

`<experiment>` is one of `fold_1`, `fold_2`, `fold_3` or `holdout`. This table lists principal folders, not every generated diagnostic file. The legacy `07_anomalies/` fallback supports archived exports; it is not a missing current pipeline stage. Do not rename artifact directories to match notebook numbers: saved pointers, input references, registrations and hashes may depend on them.

## Research and reporting scope

| Current research notebook | Former number | Intended use |
|---|---|---|
| 01 source-quality sensitivity | 09 | Historical exploratory experiment; archived combined-model checkpoint required |
| 02 matched-shop features | 10 | Historical exploratory experiment; archived combined-model checkpoint required |
| 03 longer-history experiment | 12 | Earlier history-extension study; use main 09b for the current longer-history comparison |
| 04 registration | 13 | Historical registration tool for older checkpoints; not an evaluation runner |
| 05 review readiness | 14 | Section 2: source-review log. Other sections: archived checkpoint/workload analysis |
| 06 relative-price features | 15 | Optional experiment after completed main 09b |
| 07 peer-district signals | 16 | Optional experiment after completed main 09b |
| 08 larger search budget | 17 | Optional experiment after completed main 09b; no need for further tuning merely to improve a reported score |
| 09 label/anomaly audit | 18 | Optional diagnostic after main 05/11 |
| 10 wholesale availability | 19 | Optional source-coverage audit using existing caches |
| 11 festival calendar | 20 | Optional fixed-parameter experiment after main 09b Optuna |

The main workflow supports the methods/results chapters when backed by its matching saved evidence. Research 06–11 belong in the report only where their experiments or limitations are actually discussed and their results are available. They are not required to run the dashboard. Historical research 01–03 should not be presented as results of today's four-classifier comparison. Cite research 05's source-review evidence only for cases actually recorded in the matching log; source-snapshot consistency does not settle whether a suspicious price is correct.

A fresh clone cannot reproduce the historical combined-model workflow using today's main notebooks 06–09. Restore its archived model, prepared inputs, source snapshot and matching Git revision if that history needs reproduction. Do not redirect an old model path to a current study or rewrite an old manifest to make it pass.

## November registration

The code in research 04 targets the older combined-model classifier and the earlier history-extension candidate. It records frozen copies/hashes and a proposed 2–29 November 2026 outcome period, with evaluation no earlier than 30 November. It rejects late creation of that registration. It does not establish that a registration was made, identify today's dashboard model, or execute the eventual evaluation.

No actual registration, frozen model copies or deployment artifacts are present in this cloud checkout. The claimed 20 October registration date, model AP values and cutoffs therefore remain unverified here. Verify the actual `registration.json`, both frozen hashes and the current `11_deployment/deployment_selection.json` before naming these models or comparing their results. Matching display names alone do not establish matching models.

For the current report, keep the historical registration outside the current dashboard model's performance claims unless its exact scope and results are verified. If no evaluation has been performed, report that plainly. Preserve the registration and its model copies regardless of report scope. Evaluating those frozen models would require a separate implementation following that registration; it would not automatically evaluate the current model. Do not run refresh against a reserved month to bypass this separation.

A future prospective evaluation of the current model needs a later genuinely uninspected period, a registration made before the relevant forecast/outcome exposure, and timely archived forecasts. Do not relabel the existing November registration as a prospective test of a replacement model. Previously inspected validation/test periods remain exposed.

## Execution counters and appendix wording

Saved execution counts describe kernel history, not a correctness guarantee. The standalone review cells in main 03 and research 05 can leave nonsequential counts. This review did not execute them or establish the validity of their saved outputs. Preserve the counters and outputs rather than editing the numbers or rerunning output-writing cells for appearance. A deliberate reproduction belongs in a separate run with its actual execution record.

Suggested appendix text:

> Notebook numbers identify the current presentation order, while artifact directories retain earlier workflow names to preserve saved dependencies and provenance. For example, main Notebook 12 publishes dashboard artifacts under `08_dashboard`, and archived exports may use `07_anomalies`. Research notebooks have their own numbering and include historical experiments requiring archived checkpoints. Nonsequential saved execution counts can reflect standalone diagnostic cells; they are not evidence of a fresh end-to-end execution. The artifact map distinguishes current outputs from archived dependencies.

Report the real verification separately: static checks, synthetic tests, saved-artifact compatibility checks and real-data execution answer different questions. Do not describe this documentation update as an error-free end-to-end run.
