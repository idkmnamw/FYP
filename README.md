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

The four classifier studies use 16 sequential Optuna trials, including reference configurations. Repeated exact settings reuse their three fold metrics within the same run instead of refitting. The trial table records `reused_from_trial` and zero additional fitting time; the nominal budget is 16 trials, not necessarily 16 distinct configurations. The sampler, objective and earliest-trial tie rule are unchanged. Select by mean AP on three expanding 2025 folds. Preprocessing and class weights are fitted using training labels only. Fit selected settings through December 2025, then choose the cutoff on January–March 2026 validation. Different score cutoffs do not imply equal workloads; cutoff-selection metrics are not independent test estimates.

**Active longer-history stage:** run `05b_historical_data_audit.ipynb` once if a compatible 2024 audit is not already saved. Its source-cache directories are retained, so existing verified 2024 sources can be reused. After 09, run 09b and then 10. The two history candidates inherit 09's original 31-leaf reference and its selected settings; there is no new parameter search. Each adds 2024 to every fold's training history while retaining that fold's original basket eligibility and unchanged validation features/labels/rows. Historical metadata/units remain provisional; exclusions and training support are audited. Outputs are separate under `09b_longer_history_lightgbm/<timestamp>/{reference,tuned}/`.

10 verifies the same parent 09 study, historical/cleaning hashes, exact prepared-input hashes and matching validation row identities/labels, then identifies the candidate under its declared fold-AP rule. It does not select by later validation, retrain a model, change thresholds, read prepared test rows or replace a dashboard checkpoint. Report the different search spaces, nominal trial budgets and runtime.

The old combined training notebook and duplicate research model notebooks have been removed from the active tree. Exact previous source and executed outputs remain in Git history (for example commit `7cbd4ec`), and local result folders are untouched. Saved model notebook outputs describe the last local execution; your completed model studies and results are preserved. Their dated report notes, where present, describe earlier saved experiments; update them when quoting a new run.

## Existing dashboard and later stages

| Notebook | Purpose |
|---|---|
| `11_anomaly_detection.ipynb` | Pin 10’s candidate/cutoff and fit or reuse a separate 2025-reference Isolation Forest |
| `12_shap_and_dashboard_inputs.ipynb` | Verify the pinned candidate, precompute SHAP, and publish its own metrics/group evidence |
| `13_refresh_dashboard.ipynb` | Refresh observations using the frozen dashboard model |

Renumbering does not change the existing artifact directories `06_models`, `07_anomalies` or `08_dashboard`. The dashboard continues using its verified saved checkpoint, independently of the retired source notebook. Opening the app requires no notebook rerun. Anomaly flags are not confirmed errors or supervised surge labels. SHAP describes model contributions, not economic causes.

**Deployment workflow:** once 10 is complete, run **11 → 12 → 13**. No legacy combined-training checkpoint is required. 11 freezes 10's candidate and cutoff in `11_deployment/deployment_selection.json`; later comparisons do not silently replace it. A compatible original anomaly detector is reused by exact training hash/formulas/settings/version, otherwise a separate training-only detector is fitted. It is not retrained merely because the classifier changed.

12 publishes a separate `08_dashboard/deployments/<id>/<export>/` export, then writes `08_dashboard/active_deployment.json` **last**, after score consistency, SHAP additivity and hash checks pass. The app follows that pointer and displays the selected candidate's own validation metrics and group guidance. Old exports/refreshes stay intact, and failed builds leave the old dashboard active. 13 refreshes the active model without fitting or changing the cutoff. Derived explanations and table reads are offline; the app never fits a model or computes SHAP.

For normal use, open the app. To update observations, run **13 only**. To verify/rebuild the same selected-model handoff, run **11 then 12**, followed by 13 if needed; do not restart 01–10. Pinning a future replacement needs an explicit new deployment decision, not editing hashes. Current validation/cutoff evidence remains exploratory and is not an untouched test. The existing November registration refers to its recorded checkpoints; deploying a new model does not amend that protocol or establish timely archived forecasts for it.

The app provides local filters, price stories/history, alerts, unusual patterns, cached explanations and downloads. The optional shortlist limits review workload, not model scores; omitted alerts may contain surges. Dates, units and coverage remain visible. Some rows have no cached explanation.

13 refreshes closed calendar weeks without training or changing the cutoff. It shows observation/scoring dates and coverage audits. The latest saved Notebook 13 execution supports **5,357 local rows for 19–25 October 2026**, versus **5,343** in the original June snapshot; this is coverage, not accuracy or population coverage. Refreshed scores are retrospective offline outputs. Registered outcome months remain protected. Failed checks preserve the published pointer.

## Preparation helper

There are three Python files: `app.py`, `pricecatcher_config.py` and `prepare.py`. The notebooks hold modelling code. The helper prepares fold inputs without training a combined model:

```bash
python prepare.py --experiment all
```

Manual notebook Run All is supported. Optional `python prepare.py --rerun-all` starts a separate reproduction of **01–10**, including all four searches, longer-history fits and comparison; it excludes 11–13 and deployment. Repeated execution of exposed periods is not fresh independent validation. The old `--model-mode` option is retired.

## Supplementary baseline and reproducibility

Run **10 only** to add the food-group-rate baseline to an already completed 06–09b workflow. No classifier is refitted, no new alert cutoff is chosen, and no test rows are loaded. It reads the verified prepared training labels and local 2024 cache; no downloads occur. The 2024-added labels reuse 09b's existing weekly calculation and must match its saved training row/positive counts. Both training histories are compared on the same evaluation rows. A later comparison never replaces the already pinned dashboard model.

Read `group_baseline_metrics.csv` and `group_baseline_comparison.csv` in the new comparison folder. Positive AP differences support ranking beyond knowing the training food-group rates; negative differences must also be reported. The classifiers also use item, location and coverage information, so this comparison alone does not isolate the contribution of price history. This is supplementary analysis after earlier results were inspected, not an independent confirmation or a new selection rule. A higher fold score after tuning is not proof of better later-period performance. The current tuned model's many false alerts and weak group support remain limitations.

Before changing or moving your environment, activate its working virtual environment and run:

```bash
python prepare.py --export-environment
```

This writes `environment_requirements.txt` inside the active data run without running notebooks or fitting models. It checks saved deployment package versions when present, then records exact installed FYP package versions and the Python version. If versions differ, it lists all mismatches together and prints one command to restore them; it never installs packages or edits checkpoint manifests automatically. Keep this generated file with your local run. Use the same Python version and install it in a separate virtual environment when restoring; it pins direct FYP packages, not every transitive dependency. `requirements.txt` remains the initial-install guide, not the exact checkpoint environment. Do not casually upgrade the environment used for saved checkpoints.

New historical provenance uses `run::` and `repo::` references. Existing frozen manifests are not edited: their explicit legacy `data/runs/run_...` and shared `data/` suffixes can be resolved after copying. Copy the complete active run **and** shared `data/12_longer_history/` / `data/11_historical_audit/` caches when moving the repository. Original hashes still apply, and changed or missing files stop verification. Old optional research notebooks retain their original reproduction requirements.

Further parameter expansion is optional research, not a required repair. Keep the completed experiment and exposed periods intact. If another training experiment is justified, predeclare a small learning-rate/tree-count comparison and its chronological stopping/selection rules, then assess it on a genuinely untouched later period. Existing November registrations describe their recorded checkpoints; they do not automatically cover a replacement model. No additional search or deployment switch is made by these maintenance changes.

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
- **Missing dashboard exports:** run 11 then 12 to verify the selected checkpoint, or verify an existing refresh with 13. Do not restart 01 just to open the app.
- **Downloads:** official source access is needed for uncached data. A verified complete cache avoids downloads; this cloud has previously returned proxy 403. Partial downloads are not accepted.
- **Caches:** `.venv/` is the working environment; notebook checkpoints/Python caches can be recreated. Do not delete generated model/source evidence while cleaning files.

The active workflow now evaluates added history before model comparison. The older historical experiment is retained only as provenance for the current dashboard; its scores are not automatically assigned to 09b. Comparing six candidates is an exploratory extension after prior outcomes were inspected, not independent confirmation.

## Reading model limitations

The app's How it works → Model results and research details section shows the pinned candidate, its exact cutoff, the validation period and actual group support/detection. Story text and CSV guidance use that same model's group evidence. A missing or below-cutoff alert does not imply stable prices. Vegetables may dominate correct alerts; chicken, eggs and flour can miss most surges, while oil/rice/sugar have too few positive cases for broad claims. False alerts remain review work, not automatic consumer or policy warnings. Model integration does not repair unresolved source prices, add external drivers or prove affordability benefits.

To browse the preserved legacy dashboard intentionally, stop Streamlit and back up/remove only `08_dashboard/active_deployment.json`, then restart. Do not delete models, data or manifests inside preserved caches. A pending new pin still prevents refresh of a different model by accident; finish the intended deployment or make a documented rollback decision before refreshing.

The saved outputs in 11–13 describe the last completed local deployment/refresh. They are not a claim that the cloud independently retrained those models. No rerun of 01–10 is needed when the verified comparison and source inputs are already complete. Generated deployment/cache folders are managed automatically; ordinary use needs no folder or hash edits.

Treat generated CSV/JSON/model files as read-only evidence. Make a separate copy for report formatting instead of resaving the frozen originals.

## Price map & playback

The dashboard has three sections: **Latest saved prices**, **Price map & playback**, and **How it works**. Model scores, detailed evaluation metrics and feature explanations remain available on demand.

Open **Price map & playback**. Choose one item/unit and shop type, then press **Play** or move the weekly slider. **Pause**, **Previous**, **Next** and **First week** support manual exploration. Playback stops at the last available week; selecting a district pauses it. Click a district or use its selector, and optionally choose a second district for a shared price-history chart. Reset the sidebar's District filter to **All** for wider comparisons. Choose **Price (RM)** or **Weekly change (%)**. Percentage changes require the same local series in both exact consecutive weeks; missing weeks are not bridged. The weekly calendar includes weeks with no observations. Grey means insufficient data for the selected measure, including a missing previous week for percentage change. Unmatched district names remain in the table and selector. Latest-week model signals and cached explanations are separate from the observed-price map; historical weeks have no inferred alert.

Comparisons use sampled district medians, with record/shop/day counts and gaps visible. They do not guarantee the cheapest shop, savings or population coverage. Prices may reflect different sampled shops. Colours stay fixed across the selected item/shop/filter history during playback; percentage-change colours use a symmetric scale around zero. Changing item, shop type or filters may change the scale. Playback uses a timed Streamlit fragment to update only the map section; browsing never recomputes SHAP or model predictions. No model, cutoff, prepared data or notebook outputs are changed, and browsing requires no map service or model fitting.

The bundled [district boundaries](assets/malaysia-districts.geojson) are the unmodified 160-feature DOSM [`administrative_2_district.geojson`](https://github.com/dosm-malaysia/data-open/blob/21a78e98efd4cd9b022a27a1bf67d167076b7591/datasets/geodata/administrative_2_district.geojson), retrieved 28 October 2026, under DOSM's [Open Data License](assets/DOSM-OPEN-DATA-LICENSE.md). Name matching ignores case/punctuation and explicitly recognises `W.P.` / `Wilayah Persekutuan`; it does not guess district aliases or historical boundary changes. No official endorsement is implied.
