"""Prepare chronological inputs; optionally reproduce the four-model workflow."""
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
          "08_xgboost.ipynb", "09_lightgbm.ipynb", "10_model_comparison.ipynb"]


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
    args = parser.parse_args()
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
        for filename in MODELS:
            run_notebook(filename, run_root)
        print("Completed preparation and four-model comparison:", run_root)
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
