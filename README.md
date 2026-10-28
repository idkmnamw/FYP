# HargaLens — Malaysian food-price monitoring

A research prototype for weekly local prices, next-week surge review, unusual patterns and cached explanations. An alert needs investigation; scores are not verified probabilities and affordability benefits have not been demonstrated.

## Open the dashboard

From your FYP folder, activate the existing working environment:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

The dashboard has three sections: **Latest saved prices**, **Price map & playback**, and **How it works**. Opening it requires no notebook rerun. Model scores, explanations and research metrics are available on demand. Select the same environment as your VS Code notebook kernel; terminal activation does not select the kernel.

For a **fresh installation**, use the original training Python version, **3.12**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Keep a working environment until its replacement has been verified. Saved package versions matching under Python 3.14 do not establish compatibility with the original Python 3.12 training environment. On macOS, LightGBM may also need `brew install libomp`.

## Which files are needed?

| Files | Purpose |
|---|---|
| Numbered notebooks | Preparation, model studies, comparison and dashboard exports |
| `pricecatcher_config.py` | Shared settings, paths and frozen-input checks |
| `prepare.py` | Chronological fold preparation, optional reproduction and version export |
| `app.py` | Read-only Streamlit dashboard |
| `requirements.txt` | Initial dependency installation |
| `assets/` | Logo, district boundaries and their required licence |
| `research/` | Optional supporting experiments, source review and historical provenance |
| Local `data/` | Source snapshots, prepared inputs, models, manifests and exports; excluded from Git |

There are only **three Python files**. Keep `data/` and the saved evidence. `.venv*/`, Python caches and notebook checkpoints are excluded from Git; deleting them is not a way to reproduce a trained model. Treat generated model/CSV/JSON files as read-only evidence and make copies for report formatting.

## Notebook workflow

| Notebook | Purpose |
|---|---|
| 01–04 | Integrate sources, audit quality, clean and aggregate Monday–Sunday prices |
| 05 | Past-only features, fixed 10% next-week labels, chronological splits and coverage audits |
| 05b | Audit the 2024 source before adding historical training data |
| 06 | Logistic Regression |
| 07 | Random Forest |
| 08 | XGBoost |
| 09 | LightGBM |
| 09b | Compare longer-history controls and run 24 direct-history Optuna trials |
| 10 | Compare six inherited candidates plus the new history-tuned candidate, with group-rate baselines |
| 11 | Pin 10's candidate/cutoff and fit or reuse a separate training-only anomaly detector |
| 12 | Precompute SHAP and publish the pinned model's dashboard inputs |
| 13 | Refresh observations using the same frozen model and cutoff |

Use **Restart Kernel and Run All** in order. The four model notebooks can run independently after 05. Missing folds are prepared through `prepare.py`; partial or frozen inputs stop rather than being overwritten. Run 05b before 09b if no compatible audit exists. Do not run notebooks concurrently or edit `data/active_run.txt` during execution.

For an already completed project, run only the stage needed: **10** for supplementary comparisons, **11 → 12** for a model handoff, or **13** to refresh observations. Starting **01** intentionally creates a separate reproduction. Model studies save separate timestamped results; previously inspected periods remain exposed after a rerun.

Optional terminal commands:

```bash
python prepare.py --experiment all       # Prepare all chronological folds
python prepare.py --rerun-all           # Separate reproduction through 10; no deployment
python prepare.py --export-environment  # Record installed versions; no notebook execution
python prepare.py --verify-checkpoint   # Read-only Python/model/score/SHAP check
```

The environment export writes `environment_requirements.txt` inside the active run. It checks saved deployment versions and groups mismatches/missing packages into one restore command. It never installs packages automatically. Keep the file with the run; it pins direct project dependencies and records Python, not every transitive dependency. Broad `requirements.txt` ranges are not the exact checkpoint environment.

## How the experiments are evaluated

The four classifiers use **16 sequential Optuna trials** and three expanding 2025 folds. Exact repeated configurations reuse that run's fold metrics; `reused_from_trial` identifies them and additional fit time is zero. Sixteen trials need not mean sixteen distinct settings. Imputation, encoding, scaling and class weights are learned inside training folds. Mean fold average precision (AP) selects settings; January–March 2026 validation selects the alert cutoff. AP is not accuracy, and cutoff-selection performance is not an independent test estimate.

09b adds 2024 to training only, holding validation inputs/labels/rows and each fold's basket eligibility fixed. It now runs **24 sequential Optuna trials directly on the longer history**, including the two inherited settings as queued references. Learning rate (0.02–0.10), number of trees (100–400, step 50), leaves (15/31/63), minimum child samples (20/30/60/120), L2 regularisation (0–10) and class weighting (none/balanced) are searched. Mean expanding-fold AP selects parameters before January–March validation is loaded. Trials use fresh training-fitted preprocessing; exact repeats reuse fold metrics. At most 72 fold fits run, with no pruning or early stopping.

The two inherited controls still isolate the history effect. The additional `optuna` candidate tests parameter changes on that same longer history, and 10 includes it only when present in the matching run. The other classifiers have not received the same extra history. Existing saved outputs describe the earlier two-control execution; new tuning results require a local run. To run only this experiment, use the verified `.venv312` kernel and **Restart Kernel and Run All in 09b, then 10**. Do not rerun preparation or 06–09. Temporary fold training files are cleaned up after saving. Existing dashboard pins are preserved; no accuracy gain or model promotion is assumed.

10 adds training-only food-group surge-rate baselines for **both histories**. Historical labels reuse 09b's verified local cache/calculation and must match its saved training support. Read `group_baseline_metrics.csv` and `group_baseline_comparison.csv` in the new comparison folder. All six saved candidates beat their matched group-only baseline in the reviewed results. AP gains support information beyond food group, but item/location/coverage inputs also contribute; this is not a price-history-only ablation. This supplementary analysis was added after earlier outcomes were inspected.

The previously selected longer-history tuned LightGBM (before this new direct-history search) has mean fold AP **0.4247** and January–March AP **0.3189**, F1 **0.3669**, precision **28.4%**, recall **51.7%**. Its matched validation group-only AP is **0.1558**. The longer-history reference has higher validation AP (**0.3300**); the tuned candidate follows the declared fold-selection rule, not universal superiority. Original April–June test AP **0.336** and F1 **0.414** belong to an older model and cannot be assigned to this candidate.

The surge target is an observed local median increase of at least **10% next exact week**, with coverage/history rules. Missing future outcomes remain unlabelled; features use past/current data only. Notebook 05's training-only 5%/10%/15% sensitivity analysis does not prove 10% is an official or optimal economic threshold. The classifier's score cutoff is a separate choice.

## Deployment and data protection

11 pins the selected study/cutoff once in `11_deployment/deployment_selection.json`; later comparisons do not replace it. A compatible original anomaly detector may be reused by training hash, settings and version. 12 verifies scores/SHAP and publishes a separate `08_dashboard/deployments/<id>/<export>/` folder, updating `active_deployment.json` last. Failed builds leave the previous dashboard active. 13 refreshes the active model without refitting. The latest saved local refresh covers **19–25 October 2026**, with **5,357 supported rows** versus **5,343** in the June snapshot; coverage changes are not accuracy estimates.

New provenance uses `run::` / `repo::` paths. Legacy recorded data/run suffixes can be located after copying without editing frozen manifests; original hashes still apply. Copy the complete active run and shared `data/12_longer_history/` and `data/11_historical_audit/` caches when moving the project. A changed or missing input must be restored, not accepted by replacing its hash.

## Supporting research and remaining limitations

`research/09` checks source-quality sensitivity; `research/10` tests matched-shop features; historical `research/12` records the earlier history experiment; `research/14` keeps source reviews and exploratory review-workload comparisons. These preserve report evidence. Some need the original saved checkpoint and the corresponding earlier Git revision (for example `7cbd4ec`), rather than the current active workflow.

`research/13` registers outcomes **2–29 November 2026**, with evaluation no earlier than **30 November**, subject to data availability. It describes its recorded checkpoints, not an automatically substituted new model. Registration does not prove timely forecasts were archived. Notebook 13 blocks downloading a registered outcome month intentionally; keep the last valid dashboard rather than bypassing the guard.

False alerts, missed surges, vegetable-dominated positives and weak support for other groups remain substantial. The 12 shortlisted suspicious-price cases need independent evidence; statistical flags do not confirm errors and no invented corrections are applied. Sparse coverage and changing sampled shops also affect medians. Fuel, weather, imports and policy are outside the model inputs. SHAP describes model contributions, not economic causes. No stakeholder adoption, affordability improvement or inequality reduction has been established. Further tuning needs a justified separate experiment and a genuinely untouched evaluation period.

## Finish the compatibility and source checks

Keep the current environment. For the existing checkpoints, create a separate Python 3.12 environment and restore the version file already exported on your Mac:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install -r data/runs/run_20261019_013241_618240/environment_requirements.txt
python -m pip check
python prepare.py --verify-checkpoint
```

Use your actual saved run path if it differs. If Python 3.12 is unavailable, install it first. Select **`.venv312/bin/python`** in VS Code and restart kernels. The verification checks the original published snapshot: frozen hashes, recorded Python/package versions, up to 50 saved scores/alerts, cached SHAP consistency and a new five-row explanation sanity check. It does not fit models, read test labels, download data, publish exports or prove future accuracy. A pass must come from running it with your real saved files; cloud synthetic tests are not that evidence. Future model studies record Python explicitly; older checkpoints use the Python 3.12 shown in their saved outputs.

For source investigation, open **`research/14_review_readiness.ipynb` → Section 2** and run only its standalone source-review cell. It needs the existing training source context from the final review cell in 03, not the older workload checkpoint. It prints the `source_review_log.csv` path and preserves existing notes. Fill evidence reference, notes, reviewer and review date before recording a confirmed decision. Keep cases unresolved when independent evidence is unavailable. Matching the stored source or neighbouring prices cannot prove the correct value. Confirmed corrections belong in a separate documented experiment; no original prices are replaced by the review cell.

Suggested report wording, based on the saved current results:

> The longer-history tuned LightGBM was selected using mean expanding-fold average precision (0.4247). January–March 2026 validation AP was 0.3189, exceeding the history-matched food-group-rate baseline (0.1558). The decision cutoff was selected on this same validation period, so precision (28.4%), recall (51.7%) and F1 (0.3669) are apparent cutoff-selection results rather than independent test estimates. Of 7,116 alerts, 5,093 were false alerts; 1,889 of 3,912 observed surges were missed. Vegetables contributed approximately 99.6% of correctly detected validation surges, limiting generalisation across food groups. The reference longer-history model had higher validation AP (0.3300), so tuning did not improve every metric or period.

> The dashboard supports historical price exploration and prioritised human review. Refreshed observations were scored retrospectively and do not establish later-period accuracy or timely warnings. Suspicious source prices remain uncertain without independent confirmation; no causal explanations, consumer savings, affordability benefits or stakeholder adoption have been demonstrated. The older model's April–June test results and the existing registered comparison cannot be reassigned to this selected model.

For any new independent comparison, freeze the exact checkpoints, cutoffs and evaluation rules before inspecting outcomes and archive predictions before they occur. Do not amend the existing registration after exposure or bypass its download guard. No new prospective evaluation is started by these verification/review steps.

## Map and playback

Choose one item/unit and shop type, then Play/Pause or move the weekly timeline. Selecting a district pauses playback; optionally compare two district histories. Price and percentage-change views keep fixed colours across the selected history. Changes require both exact consecutive weeks; gaps are never bridged. Grey means insufficient comparable data. Unmatched districts remain in the selector/table, and historical weeks have no invented model alerts. Reset the sidebar District to **All** for wider comparisons.

Sampled medians do not guarantee the cheapest shop, savings or population coverage. The bundled [DOSM boundaries](assets/malaysia-districts.geojson) are the unmodified 160-feature [district file](https://github.com/dosm-malaysia/data-open/blob/21a78e98efd4cd9b022a27a1bf67d167076b7591/datasets/geodata/administrative_2_district.geojson), retrieved 28 October 2026, under the [Open Data License](assets/DOSM-OPEN-DATA-LICENSE.md). Matching recognises case, punctuation and equivalent `W.P.` / `WP` / `Wilayah Persekutuan` prefixes; other aliases and boundary changes are not guessed. No official endorsement is implied. Browsing needs no map service, model fitting or SHAP calculation.

## Troubleshooting

- **Environment mismatch or missing tool:** activate the intended interpreter, run `--export-environment`, follow its grouped restore command, then restart notebook kernels. Preserve the working environment and checkpoint files.
- **Missing/changed inputs:** restore the matching local snapshot. Do not delete `data/`, rewrite hashes or rerun training merely to open the dashboard.
- **Missing dashboard export:** verify the existing refresh with 13, or complete 11 → 12 for the pinned model. Keep deployment pointers and archived exports while investigating.
- **Download failure:** uncached official data needs network access. A complete verified cache avoids downloads; partial downloads are rejected. Cloud access has previously returned proxy 403.
