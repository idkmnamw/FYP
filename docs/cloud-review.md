# Cloud repository review — 14 November 2026

## Workspace and scope

The attached `/workspace/FYP` checkout started from commit `bd6d77983d8ed449073619bbb7cce2e675e675dc`, matching the remote `main` ref. The local branch was `work`. Initial Git status was clean, including untracked files. Read-only `git ls-remote origin refs/heads/main` succeeded with the platform's existing authentication.

This session reused the attached workspace and its Python 3.12.14 environment; it did not create or independently validate a newly provisioned machine. Runtime status reported running/connected, current observations and no failure. The configuration-status query succeeded. This does not diagnose the earlier readiness/capability errors or establish successful publication.

The initial check did not execute notebooks, download datasets, fit models, delete files or edit manifests. Subsequent review also executed no notebooks and fitted no models. Tests used small arrays, temporary version records and extracted calculation functions only. No data files, trained models, experiment outputs or registered evaluations were removed or replaced.

## Workflow traced

The [README dependency table](../README.md#main-workflow-and-dependencies) maps the main stages. Static source inspection covered integration, training-only screening/cleaning, complete-scope aggregation, exact-week feature/label construction, the four original classifier studies, the 2024 history extension, comparison, deployment pinning, anomaly detection, SHAP exports and refresh. All research notebooks were inventoried and their imports, prerequisite paths and relationship to the main workflow inspected; they were not independently reproduced.

Important contracts visible in code:

- 01 reuses verified integrated data/lookups or starts a separate run. Its declared source scope is 2025–June 2026.
- 02–05 learn item screening and price bounds from each experiment's training period. Fixed retrospective source exceptions apply only to holdout training by default; their review dates are unknown and remain a limitation.
- 05 reindexes each series to calendar Mondays, keeps gaps missing, and requires supported current/next-week prices and past history. The target is the observed median rise of at least 10%; future outcomes are excluded from the feature list. Splits check the next-week outcome end date against each cutoff.
- 06–09 use training-fitted preprocessing and three expanding 2025 folds. Jan–Mar 2026 selects each score cutoff and is already exposed. A file called `test.parquet` does not establish untouched data.
- 09b verifies its parent LightGBM study, adds audited 2024 history and retains the original evaluation populations. It checks reconstruction of original training inputs before comparing added-history controls and search.
- 10 verifies study/input hashes and saved metrics and writes a separate comparison. It does not replace an existing deployment pin. Its food-group baselines reconstruct training labels from existing caches without fitting classifiers.
- 11 creates a deployment pin only if absent and separately fits/reuses the anomaly detector from original 2025 training features. 12 publishes explanations and dashboard inputs for that pin.
- 13 verifies frozen inputs, versions and source identity; it checks registrations before downloading later observations and blocks entire registered outcome months. Its default date is dynamic, so a November refresh may intentionally stop. Preserve the last valid snapshot.

The `research/` registration notebook uses legacy combined-model paths. It verifies existing registrations or rejects late creation of the proposed November 2026 registration. It is not a registration stage for today's four separate classifiers. Source-only audits and optional experiments are not dashboard prerequisites. Notebook references and legacy manifest path resolution are reasons to retain these files rather than delete them during cleanup.

## Confirmed bug fixed

Notebook 08 records `xgboost_module` alongside the installed XGBoost distribution version. That key is module provenance, not an installable package. Previously:

- `prepare.py --export-environment` looked it up with distribution metadata and recommended installing the nonexistent package `xgboost_module`.
- `prepare.py --verify-checkpoint`, Notebook 12 and Notebook 13 could reject an otherwise compatible XGBoost checkpoint with `PackageNotFoundError`.

The shared `installed_package_version()` helper reads this key from `xgboost.__version__` and retains exact metadata checks for real distributions. Export restoration instructions use `xgboost-cpu` on Linux or `xgboost` elsewhere for this key. Saved manifests and expected version values remain unchanged. The original export failure was reproduced from the original Git source using temporary version records; the fixed path is covered by regression tests. End-to-end XGBoost deployment remains untested without a real saved study.

Only four version-check lines changed across Notebooks 12/13. Their saved outputs, execution counts and metadata were preserved, so those old outputs are not evidence of executing the modified code. Recorded notebook hashes were not rewritten. Reproducing a historical cache may require its original Git revision, as intended by the provenance checks.

## Dependencies and reproducibility

All top-level imported modules found by static AST inspection of the three application Python files and 26 notebooks are available in the installed environment. Dependency consistency checks pass. This does not validate every optional execution path or operating system.

The `uv`-created environment initially lacked `pip`, which is needed by repository-generated restore commands. `pip` was installed and added to the saved installation script. No dependency declarations or lockfiles changed.

Remaining limitations:

- `requirements.txt` has broad ranges and no transitive lock. A fresh installation can select different versions; it cannot establish compatibility with Mac-trained checkpoints. Use the saved Python version and per-run `environment_requirements.txt`, then verify the checkpoint. That export pins direct packages only.
- Source data can be revised. Hashes, ETags, frozen lookups and matching source caches are required for reproduction. Current lookup metadata applied to 2024 is provisional historical interpretation.
- `pricecatcher_config.py` creates preparation directories when imported. Review tests avoid this side effect by extracting the needed pure functions; CLI checks importing the configuration can create empty directories even when no model changes occur.
- Code is reused between notebooks by extracting named functions with AST. Definition checks and provenance guards help, but changes to those functions require validating dependent workflows. This review did not refactor the pipeline or duplicate it into new modules.
- Models and dashboard exports are excluded from Git. There is no `data/active_run.txt`, deployment selection, saved model, registration or real dataset in this checkout. Therefore the current deployed candidate is **unknown**. Earlier “Longer-history LightGBM (tuned)” records are not confirmation of the current pin.

## Validation evidence

| Check in this review | Result | What it establishes |
|---|---|---|
| Native Git read and initial status | Passed; clean starting checkout, remote main matches | Repository access and starting state |
| Runtime configuration status | Running/connected; query succeeded | Current instance observation only |
| Notebook schema validation and compilation | All 26 passed | JSON/notebook structure and Python syntax, without execution |
| Three application Python files | Compiled | Syntax only |
| Focused unittest suite | 8 passed | Version lookup/export/checkpoint preflight, missing-package handling, calendar gaps and 10% boundary on synthetic inputs |
| Imported-module availability | None missing | Static imports can be located in this environment |
| `uv pip check` and `python -m pip check` | Passed | Installed dependency metadata consistency |
| Saved installation script rerun | Passed | Repeatable dependency setup and kernel registration in this instance; no notebook execution |
| `git diff --check` | Passed | Patch whitespace consistency |
| Real-data execution, model scoring and SHAP checks | Not run | Blocked by absent matching artifacts |
| Full notebooks, optional studies, tuning and retraining | Not run | No claim of end-to-end reproduction |
| Real-data dashboard rendering | Not run | Missing deployment and export artifacts |

The unit suite uses no notebook runner, source requests or fitting. The version-preflight test deliberately stops before model/provenance loading; it is not a full checkpoint verification. Earlier session smoke-test claims were not used as evidence of real-data correctness here.

## Saved configuration and remaining actions

The draft read confirmed existing installation/startup instructions and the proposed `storage.data.gov.my` allowlist addition were persisted. This review saved an installation-script update to include `pip`; the backend confirmed `status: saved` and `requires_publish: true`. Draft persistence is distinct from publication and from validation in a new machine. The runtime still reports the earlier network policy; no dataset connectivity request was made in this review.

Review/save the environment settings and publish through the environment UI. Publication and fresh-machine restoration have not been verified in this task. Restore the matching Mac artifacts listed in the README before verifying the deployment pin, package compatibility, saved scores, explanations and real dashboard. Retain all old hashes and registrations; do not train a replacement merely to make those checks pass.
