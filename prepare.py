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


def export_environment():
    """Save installed FYP package versions after checking the deployed checkpoint."""
    from importlib import metadata
    import platform
    import shlex
    import re
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
    required = {}
    for record in records:
        for package, expected in record["package_versions"].items():
            if package in required and required[package] != expected:
                raise ValueError(f"Saved components require conflicting {package} versions: "
                                 f"{required[package]} and {expected}. Keep the original manifests; "
                                 "these components need compatible saved checkpoints before one environment can be exported.")
            required[package] = expected
    packages = ["pandas", "numpy", "scipy", "matplotlib", "pyarrow", "jupyterlab", "ipykernel", "nbformat", "nbclient",
                "scikit-learn", "joblib", "optuna", "lightgbm", "shap", "streamlit"]
    versions = {}
    for package in sorted(set(packages) | set(required) | {"xgboost"}):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            if package == "xgboost":
                try:
                    versions[package] = metadata.version("xgboost-cpu")
                except metadata.PackageNotFoundError:
                    pass
    # Use project constraints for missing tools that have no saved checkpoint version.
    constraints = {}
    for line in (REPOSITORY / "requirements.txt").read_text().splitlines():
        spec = line.split(";", 1)[0].strip()
        if spec and not spec.startswith("#"):
            constraints[re.split(r"[<>=!~\s\[]", spec, maxsplit=1)[0]] = spec
    mismatches, restore = [], []
    for package, expected in sorted(required.items()):
        installed = versions.get(package, "not installed")
        if installed != expected:
            mismatches.append(f"  {package}: needs {expected}; installed {installed}")
            restore.append(f"{package}=={expected}")
    for package in packages + ["xgboost"]:
        if package not in versions and package not in required:
            mismatches.append(f"  {package}: missing project dependency")
            name = "xgboost-cpu" if package == "xgboost" and platform.system() == "Linux" else package
            restore.append(constraints.get(name, name))
    if mismatches:
        command = shlex.join([sys.executable, "-m", "pip", "install", *restore])
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
    args = parser.parse_args()
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
