# HargaLens — Malaysian local food-price monitoring

A research prototype for weekly prices, next-week surge review, unusual patterns and precomputed explanations. Scores are uncalibrated; affordability benefits and adoption have not been established.

## Install and open

From the FYP folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For later dashboard visits, activate `.venv` and run the last command. Select **FYP/.venv/bin/python** as the VS Code notebook kernel. Terminal activation does not select the notebook kernel. Data and models are local, outside Git; keep `data/` and its saved checkpoints.

## One notebook per model

After completing **01–05**, use this single modelling sequence:

| Notebook | Purpose |
|---|---|
| `06_logistic_regression.ipynb` | Linear classifier, references, Optuna tuning and interpretation |
| `07_random_forest.ipynb` | Random Forest, references, Optuna tuning and interpretation |
| `08_xgboost.ipynb` | XGBoost, references, Optuna tuning and interpretation |
| `09_lightgbm.ipynb` | LightGBM, references, Optuna tuning and interpretation |
| `09b_longer_history_lightgbm.ipynb` | Add 2024 history to the reference and tuned LightGBM settings; train and evaluate both |
| `10_model_comparison.ipynb` | Verify matching studies and compare the four models plus two history candidates without refitting |

Select **Restart Kernel and Run All** in each. The models use the same prepared inputs and can run independently; finish all four and 09b before 10. Missing chronological folds are prepared safely through `prepare.py`; the combined training notebook is no longer used. Partial inputs or missing folds belonging to frozen studies stop instead of being overwritten. Keep the configuration and helper beside the notebooks.

Each model contains its technique, parameter search, chronological comparison, validation results, plots and report guidance. It saves timestamped outputs to the corresponding `06_logistic_regression/`, `07_random_forest/`, `08_xgboost/` or `09_lightgbm/` folder within the active data run. Do not run notebooks concurrently or edit `data/active_run.txt` during execution.

The four classifier studies use 16 sequential Optuna trials, including reference configurations; repeated trials are possible. Select by mean AP on three expanding 2025 folds. Preprocessing and class weights are fitted using training labels only. Fit selected settings through December 2025, then choose the cutoff on January–March 2026 validation. Different score cutoffs do not imply equal workloads; cutoff-selection metrics are not independent test estimates.

**Active longer-history stage:** run `05b_historical_data_audit.ipynb` once if a compatible 2024 audit is not already saved. Its source-cache directories are retained, so existing verified 2024 sources can be reused. After 09, run 09b and then 10. The two history candidates inherit 09's original 31-leaf reference and its selected settings; there is no new parameter search. Each adds 2024 to every fold's training history while retaining that fold's original basket eligibility and unchanged validation features/labels/rows. Historical metadata/units remain provisional; exclusions and training support are audited. Outputs are separate under `09b_longer_history_lightgbm/<timestamp>/{reference,tuned}/`.

10 verifies the same parent 09 study, historical/cleaning hashes, exact prepared-input hashes and matching validation row identities/labels, then identifies the candidate under its declared fold-AP rule. It does not select by later validation, retrain a model, change thresholds, read prepared test rows or replace a dashboard checkpoint. Report the different search spaces, nominal trial budgets and runtime.

The old combined training notebook and duplicate research model notebooks have been removed from the active tree. Exact previous source and executed outputs remain in Git history (for example commit `7cbd4ec`), and local result folders are untouched. Migrated notebooks 07–09 start with empty outputs; 06 retains your latest completed local execution. Their dated report notes, where present, describe earlier saved experiments; update them when quoting a new run.

## Existing dashboard and later stages

| Notebook | Purpose |
|---|---|
| `11_anomaly_detection.ipynb` | Training-only Isolation Forest |
| `12_shap_and_dashboard_inputs.ipynb` | Precompute selected explanations and dashboard tables |
| `13_refresh_dashboard.ipynb` | Refresh observations using the frozen dashboard model |

Renumbering does not change the existing artifact directories `06_models`, `07_anomalies` or `08_dashboard`. The dashboard continues using its verified saved checkpoint, independently of the retired source notebook. Opening the app requires no notebook rerun. Anomaly flags are not confirmed errors or supervised surge labels. SHAP describes model contributions, not economic causes.

**Current integration boundary:** 11–13 use the existing saved forecasting checkpoint and, when available, its matching longer-history extension. 10 does not deploy its candidate into those outputs. A fresh run of 01–10 alone is a model experiment, not a replacement dashboard export. On a new run without the existing checkpoint, restore the matching completed dashboard run or complete a separately reviewed integration before using 11–13. Never rename model files or substitute another experiment's metrics to bypass checks.

The app provides local filters, price stories/history, alerts, unusual patterns, cached explanations and downloads. The optional shortlist limits review workload, not model scores; omitted alerts may contain surges. Dates, units and coverage remain visible. Some rows have no cached explanation.

13 refreshes closed calendar weeks without training or changing the cutoff. It shows observation/scoring dates and coverage audits. Saved October evidence supports **2,946 local rows for 12–18 October 2026**, versus **5,343** in the original June snapshot; this is coverage, not accuracy or population coverage. Refreshed scores are retrospective offline outputs. Registered outcome months remain protected. Failed checks preserve the published pointer.

## Preparation helper

There are three Python files: `app.py`, `pricecatcher_config.py` and `prepare.py`. The notebooks hold modelling code. The helper prepares fold inputs without training a combined model:

```bash
python prepare.py --experiment all
```

Manual notebook Run All is supported. Optional `python prepare.py --rerun-all` starts a separate reproduction of **01–10**, including all four searches, longer-history fits and comparison; it excludes 11–13 and deployment. Repeated execution of exposed periods is not fresh independent validation. The old `--model-mode` option is retired.

## Supporting research and report evidence

Keep the remaining `research/09`–`research/14` notebooks as optional evidence, not part of the new model sequence:

| Research study | Purpose |
|---|---|
| 09 | Source-quality sensitivity; small mixed gains, not automatically adopted |
| 10 | Matched-shop features; original features retained |
| Historical 12 | Provenance for the older longer-history dashboard checkpoint; new active history fits are in 09b |
| 13 | Frozen future comparison registration: outcomes 2–29 November 2026; earliest evaluation 30 November, subject to data availability |
| 14 | Source-review log and exploratory review-workload comparisons |

Some research extensions require the saved original forecasting checkpoint. They do not recreate it after removal of the combined notebook; exact historic reproduction uses the corresponding earlier Git revision. Existing saved studies remain evidence.

| Earlier evidence | Result |
|---|---|
| Original April–June test | AP **0.336**, F1 **0.414**; current-change baseline AP 0.151, F1 0.225 |
| Longer-history Jan–Mar validation | AP **0.330**, F1 **0.354**, precision **23.8%**, recall **69.1%** |

These belong to different fitted models. Original test scores do not establish the longer-history checkpoint's or new tuned candidates' test performance. All previously inspected evaluation periods remain exposed. July–September exposure is uncertain. Registration alone does not establish that timely forecasts were archived.

The target is a local median increase of at least **10% next exact calendar week**, subject to coverage/history support. Missing future outcomes remain unlabelled. Notebook 05 compares 5%, 10%, 15% and a training percentile using training observations only; this supports sensitivity discussion, not an official or optimal economic threshold. Legacy labels retain their recorded comparison rule; do not mix them silently.

False alerts, vegetable-dominated positives, sparse groups, changing coverage/composition and absent fuel/weather/exchange-rate features limit conclusions. The 12 shortlisted training-price cases remain unresolved without independent evidence. Statistical flags do not confirm errors. Keep evidence-backed review decisions in the existing research/14 source-review log; no automatic corrections are applied. A new correction needs a separate documented experiment.

Research 14's validation-only 25/50/100 weekly caps increased precision but reduced recall; those results do not establish shortlist performance on new dates or dashboard filters. No universal model superiority, affordability improvement, inequality reduction or stakeholder adoption has been demonstrated.

## Troubleshooting

- **Missing package:** install requirements in the selected notebook kernel, restart and Run All. On macOS, LightGBM may require `brew install libomp`.
- **Changed or incomplete frozen inputs:** restore the matching snapshot; do not replace hashes or delete `data/`. Start 01 only for an intentional separate reproduction.
- **Missing dashboard exports:** restore the matching checkpoint and use 12, or verify an existing refresh with 13. Do not restart 01 just to open the app.
- **Downloads:** official source access is needed for uncached data. A verified complete cache avoids downloads; this cloud has previously returned proxy 403. Partial downloads are not accepted.
- **Caches:** `.venv/` is the working environment; notebook checkpoints/Python caches can be recreated. Do not delete generated model/source evidence while cleaning files.

The active workflow now evaluates added history before model comparison. The older historical experiment is retained only as provenance for the current dashboard; its scores are not automatically assigned to 09b. Comparing six candidates is an exploratory extension after prior outcomes were inspected, not independent confirmation.
