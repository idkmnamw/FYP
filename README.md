# HargaLens — PriceCatcher FYP

**See local prices clearly.** HargaLens combines Malaysian weekly food prices, next-week surge review, separate anomaly flags and precomputed explanations. It is a historical research prototype; no affordability benefit or stakeholder adoption has been demonstrated.

## Open your existing dashboard

From the FYP folder, activate your project environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the address printed by Streamlit; stop it with Ctrl+C. If you already completed 08, no notebook rerun is needed. If its dashboard manifest is missing, Run All in **08 only** after restoring/completing the matching 06 and 07 outputs. Do not restart 01 merely to reopen results.

For a first installation, run `python3 -m venv .venv` before activating it. In VS Code, choose **FYP/.venv/bin/python** as the notebook kernel. A terminal's active environment does not change the notebook kernel. JupyterLab can be started with `python -m jupyter lab`.

## What to keep

| Files | Purpose |
|---|---|
| Main notebooks 01–05 | Integration, training-only coverage screening, cleaning, calendar weeks, features/labels and audits |
| Main notebook 06 | Original chronological model comparison and frozen results |
| Main notebooks 07–08 | Separate anomaly detector, chosen-model SHAP and dashboard exports |
| Main notebook 09 | Optional offline observation refresh using the completed longer-history checkpoint; no training or evaluation |
| `app.py` | One read-only Streamlit dashboard; no model fitting or live downloads |
| `pricecatcher_config.py` | Shared preparation rules and run locations |
| `prepare.py` | Prepares missing validation folds when 06 needs them; also provides the optional terminal runner |
| `research/` | Optional studies 09–14 and their saved evidence for the report |
| `assets/hargalens-mark.png` | Shared logo and page icon |
| `requirements.txt`, `.gitignore`, this README | Installation, local-output exclusions and run instructions |

There are only three Python files. The research notebooks are not part of the daily dashboard workflow.

Generated data/models stay in ignored `data/`. Keep completed runs, source snapshots and model package versions for reproducibility. `.ipynb_checkpoints/` and `__pycache__/` can be removed when notebooks are stopped; they are recreated automatically. Keep `.venv/` for the working kernel. Do not delete `data/` to fix a hash error.

## Run a new reproduction

1. Run **01 → 05** in order with Restart Kernel and Run All. Manual 01 starts a separate run, preserving completed experiments and copying a verified local source when available.
2. Run **06**. It prepares missing expanding folds through `prepare.py`, compares models and freezes the original experiment. Reopening a completed 06 reuses its saved model.
3. Run **07 → 08**. These reuse completed checkpoints and publish derived dashboard inputs.
4. For the longer-history prototype, complete research **11 → 12** once after 06, then run **08** again. The separate 07 detector keeps its 2025 reference. No other research notebook is a prerequisite.

Do not run preparation and modelling concurrently. `data/active_run.txt` remembers the active run across kernels. Do not edit it during execution. Explicit `PRICECATCHER_DATA_DIR` and `PRICECATCHER_EXPERIMENT` overrides remain available for advanced/runner use; remove old manual override cells when using the automatic workflow.

The optional command `python prepare.py --rerun-all` runs **01–06** with fresh kernels in a separate reproduction. It does not run 07–08 or the optional studies. Repeating an exposed test period is not fresh independent validation.

## Dashboard and exported files

HargaLens means a clearer view of **harga** (price). Its basket-and-lens logo is reused from one asset. Light and dark themes are supported.

- **Overview:** filters, Spotlight price stories, food-group alert counts, the full review list and CSV export. Spotlight shows up to three highest-score cases within the filters, not confirmed surges. Its downloadable HTML brief preserves dates, model identity, coverage, explanation availability and evidence limitations; a browser can print it as a PDF.
- **Explore prices:** one item/location/shop type, price unit, 12-week/26-week/full eligible history, observation counts and cached explanations. Missing weeks remain blank; an older prediction never substitutes for an unsupported latest forecast.
- **About the model:** the displayed candidate's own validation evidence, false-alert rate, training dates, fixed cutoff, support counts and limitations.

The dashboard reads this entry point under the printed active run:

```text
08_dashboard/dashboard_manifest.json
```

It points to a model-specific cache under `08_dashboard/<model-hash>/`:

| Table | Contents |
|---|---|
| `latest_dashboard.parquet` | Supported latest rows, chosen-model scores/alerts, separate anomaly flags and explanation availability |
| `shap_explanations.parquet` | Up to 200 highest-score latest explanations and a fixed historical reference sample |
| `reference_feature_importance.csv` | Descriptive SHAP summary on up to 200 original 2025 training-input rows |

08 prefers the latest compatible completed longer-history checkpoint by completion time, never by metric ranking. It verifies checkpoint provenance before exporting that model's scores/SHAP. Without one, a new export clearly uses the original model; an already published longer-history checkpoint cannot silently disappear and downgrade the export. Original models, metrics and the old `08_explanations/` cache are preserved.

The app verifies export hashes, model/version consistency, dates, alerts and explanation mapping. It checks the frozen original train/validation/test files before reading **only price, coverage and identifier columns**, not labels, for historical charts. History is eligible original 2025–June 2026 modelling rows plus the latest snapshot, not all raw observations or the added 2024 training history. Streamlit reuses verified data across filters and invalidates its cache when relevant file metadata changes. No model object or SHAP calculation is needed while browsing.

The saved snapshot describes **22–28 June 2026**, with a forecast horizon of **29 June–5 July 2026**. These are historical dates. Scores are uncalibrated, SHAP is not causal, and the 2025 reference sample does not represent the longer model's entire training population. Missing explanations/forecasts and unsupported locations must remain explicit.

## Refresh observations without retraining

After completing 08 with the longer-history checkpoint, open **09_refresh_dashboard.ipynb** in the project kernel and **Restart Kernel and Run All**. This main-folder notebook is maintenance; `research/09_source_quality_sensitivity.ipynb` is a separate research study.

Leave `SNAPSHOT_WEEK = None` for the last calendar week fully closed before today in Malaysia, or enter an earlier Monday. The official source must contain observations through that week's Sunday. New source months are downloaded once per snapshot, processed using frozen lookups, eligibility, bounds and egg conversions, then cached and verified. Unknown/blank shop metadata is excluded and counted; statistical flags remain in the inputs. No replacement price is invented. The original June features must reconstruct before newer scores can be published.

Restart/reload the app after 09 finishes. It selects `08_dashboard/refresh_manifest.json` when present, verifies its source/cache hashes, adds refreshed eligible history and displays the actual scoring timestamp and source attrition. New explanations are computed offline for at most 200 latest cases. Browsing still requires no model loading or SHAP computation. Completed derived caches are reused; the original 08 manifest, checkpoints, prepared splits and research evidence are preserved.

These scores are a **retrospective offline demonstration**, not proof of timely forecasts or improved accuracy. Validation precision and group limitations do not change. The workflow refuses to open any monthly file overlapping a registered outcome period, including November under the current plan. Keep that comparison separate. A failed/incomplete download does not replace the published snapshot. To deliberately reopen the original June export, remove **only** `08_dashboard/refresh_manifest.json`, keep its archived caches, and restart the app.

## Optional research evidence

| Notebook in `research/` | Saved finding or purpose |
|---|---|
| 09 — source quality | Excluding 58 additional suspected training records gave small mixed validation gains; not adopted automatically |
| 10 — matched-shop features | Mean fold AP declined slightly; original feature list retained |
| 11 — historical audit | All 12 monthly 2024 files available/schema-compatible; January, July and December sampled |
| 12 — longer history | Validation AP improved 0.223 → 0.330; eggs/flour deteriorated |
| 13 — future registration | Optional frozen-model plan for outcome weeks 2–29 November 2026; no future prices/predictions/evaluation loaded |
| 14 — review readiness | Source-investigation log and validation-only weekly alert workload comparisons; no policy adoption |

Do not rerun studies merely to refresh the dashboard. Their code/input fingerprints preserve separate results; rerunning changed code may create another experiment folder. Current lookup units/geography remain a provisional interpretation of historical metadata. Study 12 retains source rows with unknown metadata in its cache and records exclusions from supplementary training in its metadata/cleaning audits.

To inspect training source prices in a completed run, run **only the final cell of 03**. Its regenerated `training_source_context.csv` is evidence, not a decision log. Study 14 preserves manual notes at `14_review_readiness/source_review/<snapshot>/source_review_log.csv`. Neighbour agreement and re-downloading the same record do not independently confirm an error; log confirmations remain user-recorded claims and never change prices/models automatically. Its 12 reviewed cases remain unresolved in the saved run.

Study 14's whole-validation workload results are:

| Policy | Average cases/week | Precision | Recall |
|---|---:|---:|---:|
| Frozen rule | 946.75 | 23.8% | 69.1% |
| At most 25/week | 25 | 56.0% | 4.3% |
| At most 50/week | 50 | 53.3% | 8.2% |
| At most 100/week | 100 | 49.5% | 15.2% |

All capped cases are vegetables. Keep the current frozen rule: these smaller lists reduce workload but miss more surges. They are exploratory workload evidence, not an accuracy improvement or metrics for arbitrary dashboard filters. No winner was selected. Study 13 remains unchanged; registration is not live evaluation. Earliest planned November evaluation is 30 November subject to data availability. July–September exposure is uncertain.

## Results and interpretation

| Evidence | Model | Appropriate comparison |
|---|---:|---:|
| Original 06 April–June test AP | 0.336 | Current-change baseline: 0.151 |
| Original 06 April–June test F1 | 0.414 | Current-change baseline: 0.225 |
| January–March validation AP | Longer history: 0.330 | Original: 0.223 |
| January–March validation F1 | Longer history: 0.354 | Original: 0.332 |

The original test has 3,985 correct alerts, 8,131 false alerts and 3,140 missed surges. It also exceeds the supplementary food-group prevalence baseline (AP 0.225; F1 0.367). This is a saved reproduction of an already exposed test period.

Longer-history validation has 2,702 correct alerts, 8,659 false alerts and 1,210 missed surges. Its precision is 23.8%; 76.2% of alerts are false. Vegetables provide 2,686 correct alerts. Sugar has no positive training/validation cases and cooking-oil positives are scarce. **Independent test performance for this candidate remains unconfirmed. Original 06 test scores do not belong to it.**

A surge is an observed local weekly median increase of **at least 10% in the next exact calendar week**, with required coverage/history. Missing outcomes stay unlabelled. Exact-week lags and past-only rolling features are used; learned screening, bounds and imputation use the appropriate training period. Original folds expand within 2025; holdout trains through December 2025, chooses its cutoff on January–March 2026 and evaluates on April–June 2026. Numerical tolerance is `1e-10` percentage points; legacy schema-2 runs retain their strict label comparison.

Sparse series, changing shop composition, unresolved suspicious prices, retrospective exclusions and temporal variability limit interpretation. 05 reports 3,271 source series without usable modelling rows and 5,343 latest supported rows. 07 flags 45 latest rows, without labelled anomaly ground truth. Medians are not fixed-shop price indices. External economic drivers are absent. No claim of universal model superiority, causal explanations, representative adoption or reduced inequality/food costs is supported.

## Common problems

- **Missing package:** select the project kernel; use `%pip install -r requirements.txt` in a temporary main-notebook cell, then restart. For a macOS LightGBM `libomp.dylib` error, run `brew install libomp` in Terminal.
- **Missing/changed frozen files:** restore the matching saved run and manifest. Do not bypass hashes, replace them with current hashes, or retrain over the completed experiment. Start 01 only for an intentional separate reproduction.
- **Source download failure:** a complete verified cache avoids downloads. Otherwise official-source network access is required. This cloud has returned proxy 403; no failed/partial download is accepted as a complete dataset.
- **Different model environment:** restore the package versions recorded with that checkpoint before loading it. Git does not contain the ignored data/models needed to run the app on another machine.
