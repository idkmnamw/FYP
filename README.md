# Localised food price surge preparation

A straightforward PriceCatcher pipeline for an FYP on next-week local food price surge risk. Run notebooks **01 → 05** in order using **Restart Kernel and Run All**. Notebook 90 is optional research evidence.

## Run

Use Python 3.11 or newer. From this repository:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows instead: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab
```

Keep the notebooks and both Python helper files together. The only shared settings are in `pricecatcher_config.py`. Outputs default to `data/` beside that file. Optionally set `PRICECATCHER_DATA_DIR` before starting Jupyter to use another location. Raw data and generated outputs are excluded from Git.

Notebook 01 downloads the selected basket from all 18 monthly source files (January 2025–June 2026); allow internet access and sufficient memory. Lookup snapshots are saved and reused. Monthly source prices may later be revised by the publisher: the source manifest records provenance but is not a frozen copy of the upstream files. Preserve your integrated Parquet and lookup snapshots for the final experiment. A failed download stops before replacing the combined output; do not continue with stale outputs from a failed run.

## Notebook responsibilities

| Notebook | Purpose | Main output |
|---|---|---|
| 01 | Integrate sources, validate joins, normalise egg units | Integrated daily Parquet and source manifest |
| 02 | Describe quality and screen item coverage using training only | Item exclusion table |
| 03 | Remove invalid/excluded records, retain statistical flags, document six suspected source exceptions | Daily Parquet and cleaning evidence |
| 04 | Aggregate complete Monday–Sunday scope weeks and build equal-premise comparator | Weekly Parquet |
| 05 | Build calendar lags, labels, chronological splits and training comparisons | Model splits, prediction inputs and settings |
| 90 (optional) | Inspect extreme training movements, matched shops and six-record sensitivity | Source diagnostic CSVs |

## What the label means

A positive label means the **observed local median** in the immediately following calendar week is at least 10% above the current week's observed median, for the same item, state, district and premise type. Prediction happens after current-week observations are available. This is a project definition, not an official economic threshold or proof that individual shops raised their prices. Daily observations weight frequently monitored shops more heavily. Changing shop composition and promotions can affect the median.

The main coverage rule remains two records and one premise in both weeks, with a previous-week price and at least two observed prices in the prior four calendar weeks. Notebook 05 compares this with two premises and three observed days, and compares observation weighting with equal shop weighting, **on training pairs only**. Comparisons do not automatically change the target. Equal weighting still allows changes in shop composition. Decide and document any change before final evaluation, then rerun preparation.

Missing weeks are inserted temporarily for exact calendar lags; prices are never filled. Partial weeks at the start/end of the data scope are excluded. Later predictions can use earlier observed history. Training outcomes end by December 2025; validation outcomes end by March 2026; test outcomes end by June 2026. Horizons crossing boundaries are excluded. Next-week coverage establishes historical label eligibility and never enters prediction features.

## Cleaning policy and limitations

Item coverage and statistical bounds are learned from training only. Borong is excluded for the retail scope. Statistical flags do not prove errors and are retained by default. Six specific training records with suspected tenfold source issues are excluded with exact-key/price checks; they are **suspected**, not externally confirmed, and no replacement prices are invented. Notebook 90 checks sensitivity to retaining them. These exceptions come from the supplied research notebooks and require the same source records. If the snapshot changes, the notebook deliberately stops for review. Other suspicious values remain a limitation. Set `REMOVE_FLAGGED_PRICES` only as a deliberate new experiment; rerun all downstream stages and document the policy.

The supplied report describes an older policy removing 1,950 flagged records and retaining partial boundary weeks. Update those sections, screenshots and all counts to match a final consistent run. Report item exclusions, label definition, coverage attrition and the uncertainty above.

## Handoff to model training

Use `05_modelling/experiment_settings.json` → `feature_columns` explicitly to build X. Never select all columns other than the label. Fit imputation, categorical encoding, scaling and any imbalance handling on training only. Treat `item_code` as a category. Use validation for model/hyperparameter/decision-threshold selection; reserve test for final reporting. Keep the fixed label threshold separate from the classifier probability cutoff.

Compare Logistic Regression, Random Forest, XGBoost and LightGBM with modest tuning, an always-no-surge baseline and a simple current-change rule. Report precision, recall, F1 and a clearly specified precision–recall metric (e.g. average precision), with per-food-group support and results. Overall performance can be dominated by vegetables. Groups with zero positives cannot support positive-class recall or a meaningful PR-AUC; report undefined metrics rather than inventing examples. Keep Isolation Forest anomaly scores separate from supervised labels and use SHAP for the selected model.

`latest_prediction_inputs.parquet` contains eligible rows for the last complete week of the configured scope, without labels or future observations. It is a historical prototype snapshot, not a live feed. Empty output is reported instead of substituting stale local weeks.

## Validation

Notebook code is supplied with cleared outputs to avoid presenting results from an older version. The development smoke check uses small synthetic monthly sources to exercise 01–05 and the optional audit, real Parquet handoffs, time boundaries and feature exclusions. This checks execution and safeguards; it does not validate the actual source data or model performance. Run the full pipeline locally before using any final research counts.
