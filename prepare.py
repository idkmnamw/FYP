"""Prepare chronological inputs; optionally reproduce classifier and longer-history studies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from importlib import reload, util

REPOSITORY = Path(__file__).resolve().parent
EXPERIMENTS = ["fold_1", "fold_2", "fold_3", "holdout"]
MODELS = ["06_logistic_regression.ipynb", "07_random_forest.ipynb",
          "08_xgboost.ipynb", "09_lightgbm.ipynb", "09b_longer_history_lightgbm.ipynb", "10_model_comparison.ipynb"]


def saved_environment_records():
    """Read verified study, detector and explanation version records without changing them."""
    import pricecatcher_config as cfg
    records = []
    pin = cfg.PROJECT_ROOT / "11_deployment" / "deployment_selection.json"
    if pin.is_file():
        selection = json.loads(pin.read_text())
        path = cfg.PROJECT_ROOT / selection["study_manifest"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != selection["study_manifest_sha256"]:
            raise ValueError("Pinned study manifest changed. Restore it before exporting the environment.")
        records.append(json.loads(path.read_text()))
        link = cfg.PROJECT_ROOT / "11_deployment" / "anomaly_selection.json"
        if link.is_file():
            detector = json.loads(link.read_text())
            path = cfg.PROJECT_ROOT / detector["manifest_path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != detector["manifest_sha256"]:
                raise ValueError("Anomaly manifest changed.")
            records.append(json.loads(path.read_text()))
        base = cfg.dashboard_directory() / "dashboard_manifest.json"
        if base.is_file():
            entry = json.loads(base.read_text())
            path = base.parent / entry["cache_dir"] / "explanation_manifest.json"
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["cache_manifest_sha256"]:
                raise ValueError("Explanation manifest changed.")
            records.append(json.loads(path.read_text()))
    return records


def export_environment():
    """Save installed FYP package versions after checking the deployed checkpoint."""
    from importlib import metadata
    import platform
    import shlex
    import re
    import pricecatcher_config as cfg
    records = saved_environment_records()
    required = {}
    for record in records:
        for package, expected in record["package_versions"].items():
            if package in required and required[package] != expected:
                raise ValueError(f"Saved components require conflicting {package} versions: "
                                 f"{required[package]} and {expected}. Keep the original manifests; "
                                 "these components need compatible saved checkpoints before one environment can be exported.")
            required[package] = expected
    # Use project constraints for missing tools that have no saved checkpoint version.
    constraints = {}
    for line in (REPOSITORY / "requirements.txt").read_text().splitlines():
        spec = line.split(";", 1)[0].strip()
        if spec and not spec.startswith("#"):
            constraints[re.split(r"[<>=!~\s\[]", spec, maxsplit=1)[0]] = spec
    packages = [name for name in constraints if name not in {"xgboost", "xgboost-cpu"}]
    versions = {}
    for package in sorted(set(packages) | set(required) | {"xgboost"}):
        try:
            versions[package] = cfg.installed_package_version(package)
        except metadata.PackageNotFoundError:
            if package == "xgboost":
                try:
                    versions[package] = metadata.version("xgboost-cpu")
                except metadata.PackageNotFoundError:
                    pass
    mismatches, restore = [], []
    for package, expected in sorted(required.items()):
        installed = versions.get(package, "not installed")
        if installed != expected:
            mismatches.append(f"  {package}: needs {expected}; installed {installed}")
            # xgboost_module is provenance, not an installable distribution.
            name = package
            if name == "xgboost_module":
                name = "xgboost-cpu" if platform.system() == "Linux" else "xgboost"
            restore.append(f"{name}=={expected}")
    for package in packages + ["xgboost"]:
        if package not in versions and package not in required:
            mismatches.append(f"  {package}: missing project dependency")
            name = "xgboost-cpu" if package == "xgboost" and platform.system() == "Linux" else package
            restore.append(constraints.get(name, name))
    if mismatches:
        command = shlex.join([sys.executable, "-m", "pip", "install", *dict.fromkeys(restore)])
        raise ValueError("Checkpoint versions or project packages need attention:\n" + "\n".join(mismatches)
                         + "\n\nRestore all listed packages together in this environment:\n" + command
                         + "\n\nUse the Python interpreter used for training. Restart open notebook kernels, "
                         "then rerun --export-environment. No models or manifests have been changed.")
    lines = [f"# Python {platform.python_version()}; FYP direct packages, not a complete transitive lock.",
             "# Exported from the installed environment; saved deployment versions checked when present."]
    lines += [f"{package}=={versions[package]}" for package in packages]
    version = versions["xgboost"]
    lines += [f'xgboost-cpu=={version}; platform_system == "Linux"',
              f'xgboost=={version}; platform_system != "Linux"']
    path = cfg.PROJECT_ROOT / "environment_requirements.txt"
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)
    print("Saved exact direct-package versions:", path)
    print("Keep the same Python version; install this file in a separate virtual environment to restore it.")


def verify_checkpoint():
    """Read-only compatibility check of the pinned classifier and published explanations."""
    import ast
    from importlib import metadata
    import platform
    import pricecatcher_config as cfg
    pin_path = cfg.PROJECT_ROOT / "11_deployment" / "deployment_selection.json"
    if not pin_path.is_file():
        raise FileNotFoundError("Complete 11 and 12 first. This check never creates a deployment pin.")
    records = saved_environment_records()
    # Older manifests omitted Python; their saved training outputs record Python 3.12.
    recorded_python = records[0].get("python_version", "3.12")
    expected_python = tuple(int(part) for part in recorded_python.split(".")[:2])
    if sys.version_info[:2] != expected_python:
        raise ValueError(f"Checkpoint training used Python {recorded_python}; this interpreter is {platform.python_version()}. "
                         "Use a separate matching environment; keep the existing environment and checkpoints.")
    problems = []
    for record in records:
        for package, expected in record["package_versions"].items():
            try:
                installed = cfg.installed_package_version(package)
            except metadata.PackageNotFoundError:
                installed = "not installed"
            if installed != expected:
                problems.append(f"{package}: needs {expected}; installed {installed}")
    if problems:
        raise ValueError("\n".join(sorted(set(problems))) + "\nUse --export-environment for grouped restore instructions.")
    pin, study_dir, study = cfg.dashboard_candidate()  # Existing pin only; exact hashes checked.
    import app
    directory = cfg.dashboard_directory()
    entry_path = app.dashboard_entry(cfg.PROJECT_ROOT)
    app.load_exports(str(cfg.PROJECT_ROOT), app.export_signature(cfg.PROJECT_ROOT))
    entry = json.loads(entry_path.read_text())
    model = entry["forecast_model"]
    if model.get("deployment_id") != pin["deployment_id"] or model["score_cutoff"] != pin["score_cutoff"]:
        raise ValueError("Published classifier/cutoff differs from the pinned selection. Verify 12 first.")
    def checked_path(root, relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Verification input points outside its saved folder.")
        return path
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    cache = checked_path(directory, entry["cache_dir"])
    manifest_path = cache / "explanation_manifest.json"
    if digest(manifest_path) != entry["cache_manifest_sha256"]:
        raise ValueError("Explanation manifest changed.")
    manifest = json.loads(manifest_path.read_text())
    if manifest["forecast_model"] != model or manifest["protocol"]["features"] != study["selection"]["feature_columns"]:
        raise ValueError("Published features/model differ from the frozen study.")
    protected = {pin_path: digest(pin_path), entry_path: digest(entry_path), manifest_path: digest(manifest_path)}
    for name, expected in manifest["output_sha256"].items():
        path = checked_path(cache, name)
        if digest(path) != expected:
            raise ValueError(f"Published cache changed: {name}")
        protected[path] = expected
    model_path = checked_path(cfg.PROJECT_ROOT, model["model_path"])
    if (model_path.parent != study_dir.resolve() or digest(model_path) != model["sha256"]
            or study["output_sha256"][model_path.name] != model["sha256"]):
        raise ValueError("Published classifier differs from its frozen study.")
    protected[model_path] = model["sha256"]
    protected[study_dir / "study_manifest.json"] = pin["study_manifest_sha256"]
    import joblib
    import numpy as np
    import pandas as pd
    from scipy.special import expit
    settings_path = cfg.PROJECT_ROOT / "experiments" / "holdout" / "05_modelling" / "experiment_settings.json"
    settings_hash = study["input_sha256"][str(settings_path.relative_to(cfg.PROJECT_ROOT))]
    if digest(settings_path) != settings_hash:
        raise ValueError("Frozen feature settings changed.")
    protected[settings_path] = settings_hash
    settings = json.loads(settings_path.read_text())
    features = settings["feature_columns"]
    columns = list(dict.fromkeys(features + ["forecast_id", "model_score", "predicted_surge"]))
    latest = pd.read_parquet(cache / "latest_dashboard.parquet", columns=columns)
    if latest.empty or not latest.forecast_id.is_unique:
        raise ValueError("Verification requires nonempty, unique saved forecast rows.")
    # Load calculation definitions only: never execute notebook training or publishing cells.
    notebook = json.loads((REPOSITORY / "12_shap_and_dashboard_inputs.ipynb").read_text())
    names = {"model_inputs", "dense_matrix", "source_features", "explain_rows"}
    functions = [node for cell in notebook["cells"] if cell["cell_type"] == "code"
                 for node in ast.parse("".join(cell["source"])).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(functions) != len(names) or {node.name for node in functions} != names:
        raise ValueError("Expected explanation calculations are missing or duplicated.")
    namespace = {"np": np, "pd": pd, "expit": expit}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "saved explanation calculations", "exec"), namespace)
    pipeline = joblib.load(model_path)
    sample = latest.head(50)
    scores = pipeline.predict_proba(namespace["model_inputs"](sample, settings))[:, 1]
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Reloaded classifier produced invalid scores.")
    np.testing.assert_allclose(scores, sample.model_score, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(scores >= pin["score_cutoff"], sample.predicted_surge)
    explanations = pd.read_parquet(cache / "shap_explanations.parquet")
    if explanations.empty:
        raise ValueError("The published explanation cache is empty.")
    shap_columns = ["shap__" + name for name in features]
    if not np.isfinite(explanations[shap_columns + ["base_value", "raw_output", "model_score"]].to_numpy(dtype=float)).all():
        raise ValueError("Cached explanations contain non-finite values.")
    np.testing.assert_allclose(explanations.base_value + explanations[shap_columns].sum(axis=1),
                               explanations.raw_output, rtol=1e-5, atol=1e-5)
    linked = explanations.raw_output if model["family"] == "Random Forest" else expit(explanations.raw_output)
    np.testing.assert_allclose(linked, explanations.model_score, rtol=1e-5, atol=1e-6)
    local = explanations.loc[explanations.cohort.eq("latest")].set_index("explanation_id")
    if not local.index.is_unique or not set(local.index).issubset(set(latest.forecast_id)):
        raise ValueError("Cached explanations do not match saved forecasts.")
    if manifest["protocol"].get("latest_rows") == "all" and set(local.index) != set(latest.forecast_id):
        raise ValueError("Full-coverage export has missing explanations.")
    np.testing.assert_allclose(latest.set_index("forecast_id").loc[local.index, "model_score"], local.model_score,
                               rtol=1e-6, atol=1e-8)
    # New five-row SHAP sanity check; its background is for verification only, never published.
    namespace["explain_rows"](pipeline, model["family"], sample.head(5), sample, settings)
    if any(digest(path) != expected for path, expected in protected.items()):
        raise ValueError("A frozen verification input changed during the check.")
    print("PASS Python:", platform.python_version(), "|", sys.executable)
    print("PASS pinned classifier:", pin["candidate"], "| cutoff:", pin["score_cutoff"])
    print("Snapshot checked:", entry.get("latest_complete_week", "saved export"), "(current published snapshot)")
    print(f"Cached latest SHAP coverage: {len(local):,} / {len(latest):,}")
    print(f"PASS {len(sample)} saved scores/alerts, cached SHAP consistency and {min(5, len(sample))} fresh explanations")
    print("No fitting, downloads, test-label reads, source corrections or dashboard changes were performed.")
    print("This checks technical compatibility, not future accuracy or source-price correctness.")


def expand_shap_cache():
    """Explain every row of the already published snapshot; never refresh prices."""
    import ast
    import shutil
    from time import perf_counter
    import joblib
    import numpy as np
    import pandas as pd
    from scipy.special import expit
    import app

    run = app.active_run().resolve()
    pin_path = run / "11_deployment" / "deployment_selection.json"
    if not pin_path.is_file():
        raise FileNotFoundError("Restore the existing deployment pin; expansion never creates one.")
    import pricecatcher_config as cfg
    if cfg.PROJECT_ROOT.resolve() != run:
        raise ValueError("Dashboard and configuration identify different active runs.")
    pin, study_dir, study = cfg.dashboard_candidate()
    entry_path = app.dashboard_entry(run)
    previous_entry = entry_path.read_bytes()
    entry, manifest, latest, explanations, _, _ = app.load_exports(str(run), app.export_signature(run))
    source_cache = app.inside(entry_path.parent, entry["cache_dir"])
    for name, expected in manifest["output_sha256"].items():
        if app.file_hash(app.inside(source_cache, name)) != expected:
            raise ValueError("A saved explanation-cache artifact changed.")
    model = entry["forecast_model"]
    if model.get("deployment_id") != pin["deployment_id"] or model["score_cutoff"] != pin["score_cutoff"]:
        raise ValueError("Publish the pinned deployment before expanding its explanations.")
    model_path = app.inside(run, model["model_path"])
    if model_path.parent != study_dir.resolve() or app.file_hash(model_path) != model["sha256"]:
        raise ValueError("The published classifier differs from the pinned study.")
    for record in [study, manifest]:
        for package, expected in record.get("package_versions", {}).items():
            if cfg.installed_package_version(package) != expected:
                raise ValueError(f"Use the saved {package} version before expanding explanations.")
    local = explanations.loc[explanations.cohort.eq("latest")]
    if (local.explanation_id.is_unique and set(local.explanation_id) == set(latest.forecast_id)
            and latest.has_cached_explanation.all()):
        print(f"All {len(latest):,} latest rows already have verified explanations. Nothing changed.")
        return entry_path
    settings_path = run / "experiments" / "holdout" / "05_modelling" / "experiment_settings.json"
    settings = json.loads(settings_path.read_text())
    if settings["feature_columns"] != study["selection"]["feature_columns"]:
        raise ValueError("Published model features differ from the frozen study.")
    protected = {pin_path: app.file_hash(pin_path), model_path: model["sha256"],
                 study_dir / "study_manifest.json": pin["study_manifest_sha256"],
                 settings_path: study["input_sha256"][str(settings_path.relative_to(run))]}
    source_cache = app.inside(entry_path.parent, entry["cache_dir"])
    source_manifest = source_cache / "explanation_manifest.json"
    protected[source_manifest] = entry["cache_manifest_sha256"]
    files = app.TABLES + (["refresh_history.parquet"] if "refresh" in entry else [])
    for name in files:
        protected[source_cache / name] = manifest["output_sha256"][name]
    notebook = json.loads((REPOSITORY / "12_shap_and_dashboard_inputs.ipynb").read_text())
    names = {"model_inputs", "dense_matrix", "source_features", "explain_rows"}
    functions = [node for cell in notebook["cells"] if cell["cell_type"] == "code"
                 for node in ast.parse("".join(cell["source"])).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if sorted(node.name for node in functions) != sorted(names):
        raise ValueError("Missing or duplicate explanation calculations.")
    namespace = {"np": np, "pd": pd, "expit": expit}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "explanation calculations only", "exec"), namespace)
    protocol = {**manifest["protocol"], "schema_version": 3, "latest_rows": "all",
                "maximum_batch_rows": 128, "maximum_dense_cells": 2_000_000}
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    protocol["calculation_sha256"] = hashlib.sha256(code.encode()).hexdigest()
    family = study["selection"]["family"]
    background = pd.DataFrame(columns=settings["feature_columns"])
    if family == "Logistic Regression":
        train_path = settings_path.parent / "train.parquet"
        protected[train_path] = study["input_sha256"][str(train_path.relative_to(run))]
        if app.file_hash(train_path) != protected[train_path]:
            raise ValueError("Frozen training reference changed.")
        training = pd.read_parquet(train_path, columns=settings["feature_columns"])
        background = training.sample(min(protocol.get("background_rows", 100), len(training)),
                                     random_state=protocol.get("seed", 42))
    started = perf_counter()
    pipeline = joblib.load(model_path)
    values, base, raw, scores, _ = namespace["explain_rows"](pipeline, family, latest, background, settings)
    np.testing.assert_allclose(scores, latest.model_score, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(scores >= pin["score_cutoff"], latest.predicted_surge)
    identifiers = list(dict.fromkeys(cfg.GROUP_KEYS + ["week_start", "next_week_end"]))
    full = latest[identifiers].reset_index(drop=True).copy()
    full["cohort"], full["explanation_id"] = "latest", latest.forecast_id.to_numpy()
    full["base_value"], full["raw_output"], full["model_score"] = base, raw, scores
    shap_columns = ["shap__" + name for name in settings["feature_columns"]]
    full[shap_columns] = values
    combined = pd.concat([explanations.loc[explanations.cohort.eq("historical_reference")], full], ignore_index=True)
    if not combined.explanation_id.is_unique or not np.isfinite(combined[shap_columns + ["base_value", "raw_output", "model_score"]].to_numpy(dtype=float)).all():
        raise ValueError("Invalid full-coverage explanations; previous export preserved.")
    np.testing.assert_allclose(full.base_value + full[shap_columns].sum(axis=1), full.raw_output, rtol=1e-5, atol=1e-5)
    updated = latest.copy()
    updated["has_cached_explanation"] = True
    cache_id = hashlib.sha256(json.dumps({"source": entry["cache_manifest_sha256"], "protocol": protocol}, sort_keys=True).encode()).hexdigest()[:16]
    target = entry_path.parent / f"full_shap_{cache_id}"
    if target.exists():
        raise ValueError(f"A full-SHAP cache already exists at {target}; preserve and inspect it. Nothing overwritten.")
    def check_inputs():
        for path, expected in protected.items():
            if app.file_hash(path) != expected:
                raise ValueError(f"Frozen input changed: {path}. Previous export preserved.")
        if entry_path.read_bytes() != previous_entry:
            raise ValueError("Dashboard publication changed during expansion.")
    check_inputs()
    target.mkdir()
    updated.to_parquet(target / "latest_dashboard.parquet", index=False)
    combined.to_parquet(target / "shap_explanations.parquet", index=False)
    for name in files:
        if name not in {"latest_dashboard.parquet", "shap_explanations.parquet"}:
            shutil.copyfile(source_cache / name, target / name)
    (target / "previous_entry.json").write_bytes(previous_entry)
    output_hashes = {name: app.file_hash(target / name) for name in files + ["previous_entry.json"]}
    expanded = {**manifest, "protocol": protocol, "latest_explained_rows": len(latest),
                "offline_cache_build_seconds": perf_counter() - started, "output_sha256": output_hashes,
                "coverage_expansion": {"previous_entry": str((target / "previous_entry.json").relative_to(run)),
                                       "source_cache_manifest_sha256": entry["cache_manifest_sha256"],
                                       "generated_at": pd.Timestamp.now(tz="Asia/Kuala_Lumpur").isoformat(),
                                       "prices_refreshed": False, "model_fitted": False}}
    expanded["limitations"] = [value for value in manifest.get("limitations", [])
                               if value not in {"Prioritised latest subset", "All supported latest rows explained; historical reference remains sampled"}] + ["All supported latest rows explained; historical reference remains sampled"]
    (target / "explanation_manifest.json").write_text(json.dumps(expanded, indent=2))
    published = {**entry, "cache_dir": target.name,
                 "cache_manifest_sha256": app.file_hash(target / "explanation_manifest.json"),
                 "limitations": expanded["limitations"],
                 "files": {name: str(Path(target.name) / name) for name in files}}
    # app.load_exports adds display-only audit fields; retain the original publication schema.
    for key in ["refresh_audit", "original_supported_rows"]:
        if key not in json.loads(previous_entry):
            published.pop(key, None)
    check_inputs()
    temporary = entry_path.with_suffix(".full-shap.tmp")
    temporary.write_text(json.dumps(published, indent=2))
    temporary.replace(entry_path)
    try:
        app.load_exports(str(run), app.export_signature(run))
    except Exception:
        temporary.write_bytes(previous_entry)
        temporary.replace(entry_path)
        raise
    print(f"Published full SHAP coverage: {len(latest):,} / {len(latest):,} latest rows.")
    print("Same observation week, model, scores, cutoff and anomaly flags. No source downloads or test-label reads.")
    print("Full SHAP cache:", target)
    print("Previous cache retained; previous entry archived at:", target / "previous_entry.json")
    return entry_path


def run_notebook(filename, run_root):
    import nbformat
    from nbclient import NotebookClient
    kernel_root = run_root / "jupyter"
    kernel = kernel_root / "kernels" / "pricecatcher"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "PriceCatcher", "language": "python"}))
    os.environ["JUPYTER_PATH"] = str(kernel_root) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    notebook = nbformat.read(REPOSITORY / filename, as_version=4)
    print("Running:", filename, flush=True)
    try:
        NotebookClient(notebook, timeout=None, kernel_name="pricecatcher",
                       resources={"metadata": {"path": str(REPOSITORY)}}).execute()
    finally:
        output = run_root / "executed"
        output.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS + ["all"], default="holdout")
    parser.add_argument("--refresh", action="store_true", help="Rebuild preparation only when no frozen model exists")
    parser.add_argument("--rerun-all", action="store_true", help="Run 01–10 in a separate run; excludes dashboard stages")
    parser.add_argument("--export-environment", action="store_true", help="Save exact installed FYP package versions; no notebook execution")
    parser.add_argument("--verify-checkpoint", action="store_true", help="Read-only model/score/SHAP compatibility check; no fitting or publishing")
    parser.add_argument("--expand-shap", action="store_true", help="Explain all rows of the current saved dashboard snapshot; no repricing or training")
    args = parser.parse_args()
    if args.expand_shap:
        if args.verify_checkpoint or args.export_environment or args.refresh or args.rerun_all or args.experiment != "holdout":
            parser.error("Use --expand-shap by itself.")
        expand_shap_cache()
        return
    if args.verify_checkpoint:
        if args.export_environment or args.refresh or args.rerun_all or args.experiment != "holdout":
            parser.error("Use --verify-checkpoint by itself.")
        verify_checkpoint()
        return
    if args.export_environment:
        if args.refresh or args.rerun_all or args.experiment != "holdout":
            parser.error("Use --export-environment by itself.")
        export_environment()
        return
    missing = [name for name in ["nbformat", "nbclient", "ipykernel"] if util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)} in {sys.executable}. Install requirements.txt in this interpreter.")
    if args.rerun_all:
        if args.refresh or args.experiment != "holdout":
            parser.error("Use --rerun-all by itself.")
        import pricecatcher_config as cfg
        base = Path(os.environ.get("PRICECATCHER_DATA_DIR") or REPOSITORY / "data").expanduser().resolve()
        run_root = cfg.create_run_folder(base)
        if not os.environ.get("PRICECATCHER_DATA_DIR"):
            cfg.remember_run(run_root)
        env = {**os.environ, "PRICECATCHER_DATA_DIR": str(run_root), "PRICECATCHER_EXPERIMENT": "holdout"}
        print("New reproduction:", run_root, flush=True)
        print("Earlier evaluation periods remain exposed; this is not a new independent test.", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--experiment", "all"],
                       cwd=REPOSITORY, env=env, check=True)
        os.environ.update({k: env[k] for k in ["PRICECATCHER_DATA_DIR", "PRICECATCHER_EXPERIMENT"]})
        if not (cfg.DATA_HOME / "11_historical_audit" / "monthly_availability.csv").is_file():
            run_notebook("05b_historical_data_audit.ipynb", run_root)
        for filename in MODELS:
            run_notebook(filename, run_root)
        print("Completed preparation, classifier/history studies and comparison:", run_root)
        return
    if args.experiment == "all":
        for name in EXPERIMENTS:
            command = [sys.executable, str(Path(__file__).resolve()), "--experiment", name]
            if args.refresh:
                command.append("--refresh")
            subprocess.run(command, cwd=REPOSITORY, check=True)
        return
    os.environ["PRICECATCHER_EXPERIMENT"] = args.experiment
    import pricecatcher_config as cfg
    cfg = reload(cfg)
    required = [cfg.STAGE5_DIR / name for name in ["experiment_settings.json", "train.parquet", "validation.parquet",
                "latest_prediction_inputs.parquet"]] + [cfg.STAGE4_WEEKLY_PARQUET, cfg.STAGE4_DIR / "preparation_settings.json"]
    if cfg.TEST_END_DATE:
        required.append(cfg.STAGE5_DIR / "test.parquet")
    complete = all(path.is_file() for path in required)
    if complete:
        settings = json.loads((cfg.STAGE5_DIR / "experiment_settings.json").read_text())
        complete = all(settings["preparation"].get(k) == v for k, v in cfg.PREPARATION_SETTINGS.items())
        manifest = cfg.STAGE1_DIR / "source_manifest.json"
        complete = (complete and manifest.is_file() and settings["preparation"].get("source_manifest_sha256")
                    == hashlib.sha256(manifest.read_bytes()).hexdigest())
    if complete and not args.refresh:
        print(f"{cfg.EXPERIMENT}: existing preparation reused at {cfg.RUN_DIR}")
        return
    cfg.protect_frozen_preparation()
    if cfg.STAGE1_COMBINED_PARQUET.exists() and (cfg.STAGE1_DIR / "source_manifest.json").exists():
        cfg.verify_source_snapshot()
    os.environ["PRICECATCHER_DATA_DIR"] = str(cfg.PROJECT_ROOT)
    run_notebook("01_integrate_pricecatcher.ipynb", cfg.RUN_DIR)
    for filename in ["02_data_understanding_and_quality.ipynb", "03_cleaning_and_outliers.ipynb",
                     "04_weekly_aggregation.ipynb", "05_features_and_labels.ipynb"]:
        run_notebook(filename, cfg.RUN_DIR)
    print("Prepared:", cfg.RUN_DIR)


if __name__ == "__main__":
    main()
