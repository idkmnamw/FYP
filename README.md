# HargaLens

FYP prototype for local Malaysian food-price monitoring with DOSM PriceCatcher data. It shows recorded weekly medians, next-week surge scores, unusual price patterns and saved SHAP explanations.

The target is an **observed local median increase of at least 10% in the next calendar week**. Medians reflect the shops and observations available; they are not a fixed-shop price index. Model scores prioritise review and are not calibrated probabilities.

## Open the project

Use Python **3.12**. For a development environment:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m streamlit run app.py
```

Select `.venv312/bin/python` as the VS Code notebook kernel. Run notebooks from the project root, one at a time.

**Data, fitted models and dashboard exports are excluded from Git.** Restore the matching saved run to view the dashboard; reopening it does not require retraining. `requirements.txt` gives development ranges. For a saved checkpoint, use its recorded Python/package versions and `environment_requirements.txt`. Linux uses `xgboost-cpu`; macOS uses `xgboost`, and LightGBM may need `brew install libomp`.

## Main workflow

| Notebook | Inputs | Outputs / purpose |
|---|---|---|
| 01 — Integrate | 2025–June 2026 source files or verified cache; item/premise lookups | New run, integrated data, standardised egg units and source hashes |
| 02 — Quality | Integrated data and training cutoff | Training-only item coverage and sparse-item screening |
| 03 — Clean | Integrated data and coverage | Cleaned daily prices, training-derived flags and review evidence; flags retained by default |
| 04 — Weekly prices | Cleaned daily data | Local weekly medians and coverage, excluding incomplete boundary weeks |
| 05 — Features/labels | Daily and weekly prices | Exact-week history, labels, chronological splits, latest inputs and settings |
| 05b — Historical audit | 2024 source metadata and bounded samples | Availability audit required by 09b |
| 06–09 — Classifiers | Prepared holdout and three 2025 folds | LR, RF, XGBoost and LightGBM studies; 16 Optuna trials each |
| 09b — Longer history | Completed 09, historical audit and matching 2024 cache | Reference/tuned history extensions and a 24-trial search; evaluation rows unchanged |
| 10 — Comparison | Matching studies and source caches | Verified comparisons, history-matched baselines and candidate under the fold-AP rule |
| 11 — Anomalies/deployment | Comparison or existing deployment pin; 2025 training features | Frozen Isolation Forest and scores; classifier pin created only if absent |
| 12 — SHAP/export | Pinned models, prepared history and latest inputs | Local/global explanations and dashboard export |
| 13 — Optional refresh | Saved dashboard, frozen rules/models and later observations | Separate refreshed snapshot; registered outcome months are blocked |

For a new reproduction, run **01–05 → 05b → 06–09 → 09b → 10 → 11 → 12**. Classifier notebooks prepare missing expanding folds when no saved studies would be overwritten. Notebook 01 starts a separate run; later studies use timestamped or fingerprinted result folders.

For the report, follow **10: compare → 11: freeze deployment → main 12: export → research 12: retrospective test**. Main 13 refreshes observations separately.

```bash
python prepare.py --experiment all       # Prepare all chronological splits
python prepare.py --rerun-all            # Separate reproduction through 10
python prepare.py --export-environment   # Record installed package versions
python prepare.py --verify-checkpoint    # Read-only saved model/score/SHAP check
```

Uncached source downloads require `storage.data.gov.my`. A hash or source mismatch needs its matching snapshot or a separate reproduction, not edited hashes.

## Saved files

`data/active_run.txt` identifies the active `runs/run_...` folder. Keep these together:

- The complete active run, including prepared inputs, models, manifests, deployment pointers, exports, research results and registrations.
- `data/active_run.txt` and the run's environment requirements.
- Shared `data/11_historical_audit/` and `data/12_longer_history/` caches.
- Older artifacts referenced by any saved manifest or registration.

`pricecatcher_config.py` holds paths/settings and checks; `prepare.py` handles preparation and compatibility; `app.py` reads exports. Main notebooks stay at the root. `assets/` contains the logo, district boundaries and licence.

Artifact names retain earlier numbering: main 12 writes to `08_dashboard`, main 13 uses `09_refresh`, and old exports may reference `07_anomalies` or `06_models`. The VS Code settings hide some generated folders; hidden does not mean safe to delete. Check references before removing or renaming files.

Input references use `run::` and `repo::`, with legacy paths resolved by the config module. Saved hashes remain authoritative. Keep the matching Git revision for exact historical reproduction. Some notebook files are fingerprinted as whole files, so even markdown edits can change provenance on a future run; the cleanup leaves the shared 09b/12 helper notebooks unchanged.

## Research notebooks

Research numbering is independent of the main workflow. These experiments keep deployment separate.

| Research | Scope / prerequisites |
|---|---|
| 01–03 | Archived source-quality, matched-shop and history experiments; need the original combined-model checkpoint and matching revision |
| 04 | Registration tool for older models, not an evaluation runner; preserve registrations and verify their exact model identities |
| 05 | Standalone source review in section 2; other sections need the archived model |
| 06–08 | Relative-price, cross-district and larger-search experiments using completed 09b outputs |
| 09 | Label/aggregation and anomaly audit; requires 05 and 11 |
| 10 | Wholesale availability audit of existing training caches; no fitting or downloads |
| 11 | Festival-calendar experiment using the 09b Optuna control |
| 12 | Frozen-model April–June retrospective test, baselines and label audit |
| 13 | Shop-composition experiment using the pinned tuned configuration and 2024-added training; six research fits |

**Research 12:** Restart Kernel → Run All with the saved pin, model, test file and training-history artifacts. It saves predictions, overall/weekly/group metrics, confusion counts, baseline comparisons and a limited label audit in `research/12_test_set_evaluation/<id>/` within the active run. Baselines are no surge, current change and history-matched training food-group rates. Unseen groups use the overall training rate. Food-group AP is ranking only; no cutoff is chosen. Changed calculations produce separate results; matching completed results are verified and reused.

**Research 13:** Restart Kernel → Run All to compare original features with matched-shop count, returning-shop share and matched-shop price change. Each fold uses its verified frozen settings. Missing previous-week comparisons stay missing; observed zero overlap stays zero. Both fits use identical rows/labels. The rule recorded before fitting requires higher mean AP and no fold decline beyond 0.01. Results are exploratory and saved in `research/13_shop_composition/<id>/`; no later period is scored and no deployment is changed.

<details>
<summary>Earlier research filenames</summary>

| Old number | Current file |
|---|---|
| 09 | [01_source_quality_sensitivity.ipynb](research/01_source_quality_sensitivity.ipynb) |
| 10 | [02_matched_shop_feature_experiment.ipynb](research/02_matched_shop_feature_experiment.ipynb) |
| 12 | [03_longer_history_experiment.ipynb](research/03_longer_history_experiment.ipynb) |
| 13 | [04_register_future_evaluation.ipynb](research/04_register_future_evaluation.ipynb) |
| 14 | [05_review_readiness.ipynb](research/05_review_readiness.ipynb) |
| 15 | [06_relative_price_features.ipynb](research/06_relative_price_features.ipynb) |
| 16 | [07_cross_district_signals.ipynb](research/07_cross_district_signals.ipynb) |
| 17 | [08_larger_optuna_budget.ipynb](research/08_larger_optuna_budget.ipynb) |
| 18 | [09_label_and_anomaly_audit.ipynb](research/09_label_and_anomaly_audit.ipynb) |
| 19 | [10_wholesale_availability_audit.ipynb](research/10_wholesale_availability_audit.ipynb) |
| 20 | [11_festival_calendar_experiment.ipynb](research/11_festival_calendar_experiment.ipynb) |

Legacy callers resolve through `RESEARCH_NOTEBOOK_NAMES`; saved output directories keep their original names.
</details>

## Dashboard

- **Latest saved prices:** top cases with saved SHAP summaries, watchlist, anomaly details and downloadable case briefs. The opening headline counts alerts within active filters and shows monitoring support; it is not a population-risk ranking.
- **Price map & playback:** select item/unit and shop type, then play weekly observed medians or compare districts. Grey means insufficient comparable mapped data. Reset District to All for wider comparisons. Supported Streamlit versions allow the headline button to open this tab.
- **Understand a prediction:** separate observed prices/changes from next-week signals and inspect signed SHAP contributions. Missing explanations remain unavailable.
- **How it works:** verified retrospective test metrics and AP baselines, reference-sample global SHAP, model details and interpretation notes. Test results must match the model, deployment, cutoff and test hash; displayed CSV hashes are checked.

For full local SHAP on an existing snapshot, run **main 12** with `EXPORT_MODE = "auto"`. It expands/verifies that saved week while retaining prices, scores, cutoff and prior caches. First exports or newly pinned deployments use the original export route. Main 13 is only needed for newer observations.

Both export notebooks explain all supported latest cases in bounded batches. The **200 historical reference rows** and **100-row linear-model background** serve different purposes and do not cap latest-case coverage. Check the printed actual N/N count and the dashboard's test panel on your local snapshot. Presentation-only `app.py` changes need a dashboard restart, not notebook reruns.

## Evaluation limits and checks

- The original classifiers train on 2025; longer-history LightGBM adds 2024. Compare matching evaluation rows and input hashes.
- Three expanding folds select parameters by AP. January–March 2026 selects cutoffs; those classification metrics are apparent cutoff-selection results. April–June was previously inspected and is a retrospective test, not an untouched independent evaluation. One model's test results do not apply to another.
- Read `11_deployment/deployment_selection.json` to identify the deployed model. Later comparisons do not replace its pin. Preserve existing November registrations and outcome restrictions; a retrospective refresh is not a timely archived forecast.
- Changing shops, sparse groups and unresolved source-review cases limit interpretation. Anomaly flags are review aids; SHAP explains model output rather than economic causes. Affordability benefits and user usefulness have not been established.

`python -m pip check` checks dependency consistency. `prepare.py --verify-checkpoint` needs real saved artifacts to check compatibility. Syntax/dependency checks alone do not verify real-data execution; the repository has no standalone test suite. Saved notebook outputs document local runs, not fresh cloud executions.

If exports are missing, restore the matching run first. With the required inputs, 11–12 can build exports; avoid restarting 01 merely to reopen the app. Keep model manifests, source hashes and previous evaluations unchanged.

District boundaries are the unmodified 160-feature [DOSM source](https://github.com/dosm-malaysia/data-open/blob/21a78e98efd4cd9b022a27a1bf67d167076b7591/datasets/geodata/administrative_2_district.geojson), retrieved 28 October 2026, under the bundled [Open Data License](assets/DOSM-OPEN-DATA-LICENSE.md). See [the boundary file](assets/malaysia-districts.geojson). No official endorsement is implied.
