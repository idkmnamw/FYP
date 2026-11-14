# Research evidence and report notes

These notes preserve the previously documented local results and caveats. Research notebook filenames use the new 01–11 numbering; saved output directory names retain their original numbers. They were not reproduced or verified against real artifacts in this cloud review. See the [README](../README.md) for the current workflow.

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

## Supporting research — optional

These notebooks are outside the main run sequence. Unsuccessful experiments remain useful evidence; they do not automatically replace the dashboard model. Use the verified project kernel and run one notebook at a time.

| Notebook in `research/` | Purpose and status |
|---|---|
| `01_source_quality_sensitivity.ipynb` | Historical training-source sensitivity; requires its original checkpoint |
| `02_matched_shop_feature_experiment.ipynb` | Historical matched-shop feature comparison; requires its original checkpoint |
| `03_longer_history_experiment.ipynb` | Earlier history experiment; preserves the evidence behind the longer-history workflow |
| `04_register_future_evaluation.ipynb` | Preserve any existing future registration; this legacy tool is not a new training stage |
| `05_review_readiness.ipynb` | **Section 2 only** for the current source-review log; other sections refer to an older checkpoint |
| `06_relative_price_features.ipynb` | Seven relative-price features; no reliable overall improvement established |
| `07_cross_district_signals.ipynb` | Six lagged peer-district signals; mixed fold/validation results |
| `08_larger_optuna_budget.ipynb` | LightGBM search-budget comparison at 24/100/200 total trials |
| `09_label_and_anomaly_audit.ipynb` | Training-label sensitivity and an alternative anomaly representation |
| `10_wholesale_availability_audit.ipynb` | Source-only wholesale coverage audit; no training or downloads |
| `11_festival_calendar_experiment.ipynb` | Fixed-parameter LightGBM comparison with 12 festival-calendar flags |

Research 01–03 use the old combined-model checkpoint, not today's Logistic Regression notebook. Their original saved artifacts and matching Git revision (for example `7cbd4ec`) may be required. Keep them as historical evidence rather than running them against current classifier outputs.

Research 06–08 require completed 09b results and matching caches; 09 requires completed 05/11. They save separate outputs under the active run's `research/` folder. Do not rerun preparation to start them. Parameters are selected using expanding-fold AP; January–March results remain exploratory because that period has already been inspected.

Research 08 defaults to **200 total proposals**, including the inherited 24. Completed trials are checkpointed, so compatible progress resumes; changing 100 to 200 extends the same study. The original sampler state was not saved, so continuation is seeded but cannot reproduce that earlier state exactly. Saved mean fold AP was **0.4370 → 0.4382 → 0.4389**, and January–March AP was **0.3178 → 0.3247 → 0.3327** at 24/100/200. This is a modest recorded gain, not an independent test or evidence that AP 0.70 is attainable.

Research 09 compares label weighting and pooled Isolation Forest representations. Different labels do not establish source errors; redistributed anomaly flags do not establish improved detection accuracy. Notebook 11's pooled detector remains the dashboard component. Review `training_review_candidates.csv` with independent source evidence before drawing case-level conclusions.

The optional longer-history LR/RF/XGBoost search notebooks were removed from this workflow. Existing local research outputs can be retained for report evidence without being rerun or included in dashboard deployment. Removing their notebooks does not remove their local result folders.

For the wholesale audit, open `research/10_wholesale_availability_audit.ipynb` in the verified project kernel and **Restart Kernel and Run All**. It reads only 2024–2025 source rows from existing caches and saves five compact CSVs plus a manifest under the active run's `research/19_wholesale_availability_audit/`. Missing 2024 history is reported, not downloaded. Share `source_summary.csv` and `group_signal_coverage.csv` first. This measures potential lagged-signal coverage, not predictive accuracy; prepared labels, models and dashboard exports stay unchanged.

For the festival experiment, open `research/11_festival_calendar_experiment.ipynb` with the verified project kernel and **Restart Kernel and Run All**, after a completed 09b Optuna study. No other research notebook is required. It adds 12 before/during/after forecast-week flags for four festival anchors, with original rows, labels and model settings fixed. The calendar is embedded; no new package, download, tuning or deployment is performed. Historical calendar publication vintages are not archived, so the scheduled-date features remain a retrospective assumption. Outputs go under the active run's `research/20_festival_calendar_experiment/`. Share `fold_comparison.csv`, `validation_comparison.csv` and `calendar_feature_support.csv` first. An AP gain is not guaranteed.

## Before writing the report

The main workflow has passed isolated synthetic execution checks. The saved 30 October Notebook 10 execution also completed the comparison audit for all seven candidates while preserving the dashboard pin. These checks do not independently reproduce real-data accuracy in the cloud.

Keep these distinctions clear:

- Compare the four original classifiers on their shared 2025 training data. Present added 2024 history as a separate experiment.
- Name the model and cutoff associated with each result. Research 17's candidate and the pinned dashboard classifier are different checkpoints.
- January–March cutoff-selection metrics are exploratory. Older April–June results do not transfer to newer models.
- Preserve existing November registrations. They cover only the recorded checkpoints and rules. The proposed outcome period is **2–29 November 2026**, with evaluation no earlier than **30 November**, subject to availability. Registration had to precede 1 November; do not create a late registration and describe it as prospective. Timely forecast archiving must also be demonstrated separately.
- False alerts, missed surges, vegetable-dominated positives, sparse coverage, changing shops and provisional historical metadata remain limitations. Keep the 12 suspicious source cases unresolved unless independent evidence supports a decision.
- Fuel, weather and other external drivers are outside the inputs. SHAP shows model contributions, not economic causes. Stakeholder adoption, savings, affordability and inequality effects have not been demonstrated.
