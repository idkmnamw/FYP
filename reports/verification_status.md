# Verification and result interpretation — 18 October 2026

The repository contains source notebooks, two supporting Python files and the existing tests/reports. No new application module was added. Remote `main` was checked at `87620dd` before this update; it had no newer changes. Historical user-run results are in `completed_main_experiment.md`. They are distinct from synthetic verification and the blocked real-data reproduction below.

## Code and workflow corrections

- Preparation verifies actual integrated/lookup bytes and the combined source ID, rather than trusting the manifest alone. The runner checks sources before reusing preparation for a study without a frozen model.
- Direct runs of 01–05 refuse to replace preparation when a frozen model exists. Reproduction uses a separate `PRICECATCHER_DATA_DIR`.
- Notebook 04's coverage summary now uses the complete-scope weeks actually exported. This is a reporting correction; the modelling features/labels are unchanged.
- Before loading test rows, 06 saves the selected cutoff, input hashes and model through a recoverable checkpoint. The final model is activated only after its metadata exists. Reuse completes pending exports without fitting; completed main artifacts remain unchanged.
- A supplementary weekly audit can report different legacy/new provenance only when preparation policies agree and actual current validation prices and target labels match. The provenance difference is saved explicitly; it does not establish identical historical sources or flag assignments.
- The research question describes the combined history, item/location, coverage and seasonal model. Baseline comparisons do not isolate the effect of price history. No post-hoc target, model or cutoff tuning was introduced.
- Stale notebook outputs/execution counts were cleared. Executed copies are separate generated artifacts; historical evidence remains in this report folder and Git history.

## Verification

**16 tests passed in 113.772 seconds.** All seven notebook source files passed structure and Python syntax validation, `pip check` reported no broken requirements, and `git diff --check` passed.

The full test suite runs Notebook 01 against synthetic monthly source fixtures, and executes 02–06 plus optional 90 through the runner with fresh Jupyter kernels. It checks all expanding folds, exact calendar gaps, boundary protection, geographical attrition, training-only preprocessing, unseen categories, baseline fitting, saved predictions, source integrity and checkpoint recovery.

Additional checks cover direct-notebook protection, coverage counts matching exported weeks, provenance-only weekly audit differences, and rejection of changed audit policies/prices. Completed main files and frozen model/selection bytes are checked for preservation. These checks validate workflow behavior, not real-data accuracy.

Notebook structures/Python syntax, dependency consistency (`pip check`) and `git diff --check` were also checked. `verification_environment.txt` records the cloud environment; restoring the historical model should use its own manifest's recorded package versions.

## Full-data reproduction: blocked

The full command was attempted in `/workspace/fyp-reproduction-20261018`, separate from original artifacts:

```bash
PRICECATCHER_DATA_DIR=/workspace/fyp-reproduction-20261018 python prepare.py --experiment all --model-mode train --diagnostics
```

Notebook 01 stopped in the official lookup download with `URLError: Tunnel connection failed: 403 Forbidden`. A direct request to `https://storage.data.gov.my/pricecatcher/lookup_item.parquet` returned the same proxy denial. The failed executed notebook is retained in that reproduction folder's `01_integrated/executed/` directory. No new real-data model or metrics were produced; notebooks 02–06 could not run on real data.

The saved cloud draft lists `storage.data.gov.my` and preserves the package-manager preset, but current runtime connectivity is still denied. Apply/publish the environment configuration and reconnect as needed, then verify actual source access before resuming. A saved allowlist is not proof of runtime connectivity.

## Historical results and limits

These are archived results, not a fresh run of the revised notebooks:

| Test metric | Selected LightGBM | Current-change rule |
|---|---:|---:|
| Average precision | 0.331978 | 0.150438 |
| Precision | 0.324253 | 0.225715 |
| Recall | 0.551006 | 0.225271 |
| F1 | 0.408257 | 0.225493 |

The reported 57,456 test rows contain 7,107 positives. Confusion counts are TP 3,916, FP 8,161, FN 3,191 and TN 42,188; their precision/recall/F1 arithmetic reconciles. About 67.6% of alerts are false and 44.9% of surges are missed. False positives among actual negatives are 16.2%, a different denominator.

Vegetables contribute 89.4% of observed surges and 92.5% of detections. Chicken recall is 7.8%; oil has one positive, sugar none, and rice's reported 100% precision comes from two alerts. Overall metrics cannot establish dependable detection across every food group or geographical fairness.

LightGBM's mean fold AP of 0.354562 is close to LightGBM with 15 leaves (0.353507) and XGBoost depth 5 (0.352078). Fold AP ranges roughly 0.289–0.394, versus validation 0.222652 and test 0.331978. These are descriptive differences; causes and statistical superiority were not established. Keep the validation-selected cutoff `0.6023499784696148`; the observed test F1 at 0.5 is not grounds to change it.

An explicitly post-hoc aggregate food-group-prevalence baseline gives AP `0.2252897161`, conditional on the archived training/test group tables describing the same cohorts. Their totals reconcile, but that alone does not prove identical rows/snapshots. This baseline uses training group rates and tied test scores; no validation cutoff/F1 is inferred from unavailable validation group counts. It supports information beyond group membership, not the isolated value of temporal features.

The uploaded holdout settings could be reconstructed to match the original recorded SHA-256 exactly. Only their source-manifest hash and added source-snapshot ID differed. Restoring that file does not prove that other local prepared files or historical fold sources match; 06 still verifies their hashes. Legacy fold source consistency remains unresolved when manifests differ without comparable content IDs.

## Before the next stage

Complete a consistent-source reproduction once access works, compare source/cleaning counts and metrics with the historical evidence, and label the exposed test as reproduction/exploratory evidence. Review group/geographical support and supplementary validation diagnostics without tuning against test errors. Update the academic report's target definition, cleaning policy and limitations. SHAP precomputation, separate anomaly detection and dashboard evaluation remain next-stage work. Affordability effects, stakeholder adoption and inequality reduction have not been measured.
