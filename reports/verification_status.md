# Verification and result interpretation — 17 October 2026

This report separates three kinds of evidence. The completed user-run output snapshot is in `completed_main_experiment.md`. Tests use synthetic data. A real-data reproduction was attempted in `/workspace/fyp-reproduction-20261017`, separately from the completed experiment, but source access was blocked before integration finished.

## What is verified now

- The repository was inspected from the latest available `main` at the start of this review (`58f5b8e`).
- A fresh Python 3.12 environment installed the declared dependencies successfully.
- The existing nine-test suite passed with current packages, including real Jupyter execution on synthetic prepared folds.
- Additional tests cover content identity versus timestamped manifests and inspection of an unchanged synthetic legacy run with historical manifest differences. New training still rejects inconsistent sources, and changed frozen inputs are not accepted.
- The optional runner can execute 06 using `--model-mode reuse`, without regenerating preparation, or execute preparation plus 06 using `--experiment all --model-mode train` in a separate folder.

Final test result: **11 tests passed in 36.740 seconds**, including real-kernel initial training and command-line reuse of synthetic modern/legacy completed runs. These checks do not establish real-data performance. Only the existing two top-level Python files are needed: configuration and the optional runner.

## Full-data run: blocked, not passed

The real run attempted to load the official item lookup from `https://storage.data.gov.my/pricecatcher/lookup_item.parquet`. The network proxy returned `403 Forbidden` while opening the HTTPS tunnel. A separate direct connectivity check returned the same response. This is a cloud allowlist failure, not evidence of an invalid Parquet file or a missing Python dependency.

The cloud draft adds `storage.data.gov.my`, while preserving the existing package-manager domain presets. Review/save/publish is required to apply that draft. The source download and full 01–06 reproduction remain pending until access works. The current cloud attempt generated no new real-data model or scores. Do not describe the synthetic results as the FYP results.

## Interpretation of the historical completed outputs

These numbers are transcribed from the committed user-run output snapshot; the real dataset has not been independently rerun in this verification.

| Test metric | Selected LightGBM | Current-change rule |
|---|---:|---:|
| Average precision | 0.331978 | 0.150438 |
| Precision | 0.324253 | 0.225715 |
| Recall | 0.551006 | 0.225271 |
| F1 | 0.408257 | 0.225493 |

The selected model reports 3,916 true alerts, 8,161 false alerts and 3,191 missed surges across 57,456 test rows with 7,107 positives. Thus approximately 67.6% of issued alerts were incorrect, 44.9% of recorded surges were missed, and 16.2% of actual non-surge cases received a false alert. The first proportion is a false discovery proportion, not a false positive rate. Report these practical limitations alongside the baseline improvements.

Vegetables account for 6,354/7,107 test surges (89.4%) and 3,622/3,916 detections (92.5%). Chicken recall is only 14/179 (7.8%); cooking oil has one positive; sugar has none. Rice precision of 100% reflects only two alerts. These results support a qualified monitoring-prototype conclusion, not dependable detection across every food group.

The selected candidate's mean fold AP (0.354562) is close to LightGBM with 15 leaves (0.353507) and XGBoost depth 5 (0.352078). The differences do not establish decisive superiority. Fold AP varies from about 0.289 to 0.394; validation AP is 0.222652 and test AP 0.331978. Prevalence, time and sample composition must accompany these comparisons; the causes are not established by the metrics alone.

Keep the recorded validation-selected cutoff `0.6023499784696148`. The fixed 0.5 test F1 of 0.412746 is not a reason to change the completed experiment after inspecting its test results.

## Provenance qualification

The reported source-manifest mismatch needs more than deletion of a check. Whole manifests contain timestamps, so different hashes alone do not establish that prices changed. Conversely, unchanged model/input hashes establish that the completed artifacts were preserved; they do not prove that all historical folds used one consistent source snapshot.

New runs now use content-based source identity. Reuse of a historically inconsistent legacy manifest is explicitly recorded as unresolved provenance after exact frozen-artifact verification. Until a consistent-source real-data reproduction can run, treat cross-fold comparability as unresolved rather than asserting that the previous evaluation was fully verified.

## What must be analysed after the real reproduction

Compare the new integration/cleaning counts and source IDs with the historical evidence; report any upstream revisions. Inspect the full fold comparisons, final confusion counts, food-group/geographical support and the supplementary training-only group-prevalence baseline. Review validation quality-flag strata and group composition without treating flags as confirmed errors or the diagnostic as a retraining robustness experiment. Report differences as a reproduction/exploratory result; the exposed test period is not fresh independent evidence.

The main FYP claim should remain proportional: a baseline-comparing, explainable monitoring prototype based on observed local medians. Affordability improvements, stakeholder adoption and inequality reductions require evidence beyond these classification results. Isolation Forest, SHAP precomputation and dashboard evaluation remain separate next-stage objectives.
