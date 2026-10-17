"""Run the same readable notebooks for one experiment or all expanding folds."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo


REPOSITORY = Path(__file__).resolve().parent
EXPERIMENTS = ["fold_1", "fold_2", "fold_3", "holdout"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS + ["all"], default="holdout")
    parser.add_argument("--refresh", action="store_true", help="Regenerate preparation for a study without a saved model")
    parser.add_argument("--rerun-all", action="store_true", help="Execute all six notebooks in a new run folder, using cached source data when available")
    parser.add_argument("--model-mode", choices=["auto", "train", "reuse"], help="Run 06; auto reuses a model or creates a separate reproduction")
    args = parser.parse_args()
    if args.rerun_all:
        if args.refresh or args.model_mode is not None:
            parser.error("Use --rerun-all by itself; it handles preparation and model execution.")
        base = Path(os.environ.get("PRICECATCHER_DATA_DIR", REPOSITORY / "data")).expanduser().resolve()
        run_root = base / "runs" / datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).strftime("run_%Y%m%d_%H%M%S_%f")
        # Each new execution owns its outputs. Never overwrite an exposed run.
        candidates = [base / "01_integrated"] + sorted((base / "runs").glob("*/01_integrated"), reverse=True)
        source = next((path for path in candidates if (path / "integrated_prices.parquet").exists()
                       and (path / "source_manifest.json").exists()), None)
        run_root.mkdir(parents=True, exist_ok=False)
        if source is not None:
            shutil.copytree(source, run_root / "01_integrated", ignore=shutil.ignore_patterns("executed"))
        print(f"New reproduction: {run_root}", flush=True)
        print("This repeats an exposed test period; it is not fresh independent validation.", flush=True)
        run_env = {**os.environ, "PRICECATCHER_DATA_DIR": str(run_root)}
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--experiment", "all"],
                       env=run_env, check=True, cwd=REPOSITORY)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--model-mode", "auto"],
                       env=run_env, check=True, cwd=REPOSITORY)
        print(f"All notebooks completed. Outputs: {run_root}", flush=True)
        return
    model_only = args.model_mode in {"auto", "reuse"}
    if args.refresh and model_only:
        parser.error("--refresh regenerates preparation; use it separately from auto/reuse mode.")
    if args.experiment == "all":
        # Separate processes keep settings and notebook kernels isolated between folds.
        for experiment in ([] if model_only else EXPERIMENTS):
            command = [sys.executable, str(Path(__file__).resolve()), "--experiment", experiment]
            if args.refresh:
                command.append("--refresh")
            subprocess.run(command, check=True, cwd=REPOSITORY)
        if args.model_mode is None:
            return
        args.experiment = "holdout"

    os.environ["PRICECATCHER_EXPERIMENT"] = args.experiment
    import pricecatcher_config as cfg

    # Frozen-model reuse verifies prepared inputs in 06. New preparation also
    # checks actual source bytes, even when the runner can reuse prepared files.
    frozen_model = any((folder / "best_model.joblib").exists() for folder in cfg.MODEL_DIRS)
    if not model_only and not frozen_model:
        if cfg.STAGE1_COMBINED_PARQUET.exists() and (cfg.STAGE1_DIR / "source_manifest.json").exists():
            cfg.verify_source_snapshot()

    required = [cfg.STAGE5_DIR / name for name in ["experiment_settings.json", "train.parquet",
                "validation.parquet", "latest_prediction_inputs.parquet"]]
    if cfg.TEST_END_DATE:
        required.append(cfg.STAGE5_DIR / "test.parquet")
    required += [cfg.STAGE4_WEEKLY_PARQUET, cfg.STAGE4_DIR / "preparation_settings.json"]
    complete = all(path.exists() for path in required)
    if complete:
        settings = json.loads((cfg.STAGE5_DIR / "experiment_settings.json").read_text())
        complete = all(settings["preparation"].get(key) == value for key, value in cfg.PREPARATION_SETTINGS.items())
        manifest_path = cfg.STAGE1_DIR / "source_manifest.json"
        if manifest_path.exists():
            complete = complete and settings["preparation"]["source_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    prepare_needed = (args.refresh or not complete) and not model_only
    if prepare_needed and frozen_model:
        raise SystemExit("A completed model exists. Restore its original prepared files instead of regenerating them. Use a separate PRICECATCHER_DATA_DIR for a new study.")
    if not prepare_needed and args.model_mode is None:
        print(f"{cfg.EXPERIMENT}: existing preparation reused at {cfg.RUN_DIR}")
        return

    # Use this Python environment, even if Jupyter has a different default kernel.
    kernel_root = cfg.RUN_DIR / "jupyter"
    kernel_folder = kernel_root / "kernels" / "pricecatcher"
    kernel_folder.mkdir(parents=True, exist_ok=True)
    (kernel_folder / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "PriceCatcher", "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = str(kernel_root) + os.pathsep + os.environ.get("JUPYTER_PATH", "")

    def run_notebook(filename, output_folder):
        try:
            import nbformat
            from nbclient import NotebookClient
        except ModuleNotFoundError as error:
            raise SystemExit(f"Missing {error.name}. Activate .venv and run: python -m pip install -r requirements.txt") from error
        print(f"{cfg.EXPERIMENT}: {filename}", flush=True)
        notebook = nbformat.read(REPOSITORY / filename, as_version=4)
        if filename.startswith("06_"):
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.source = "\n".join(f'RUN_MODE = "{args.model_mode}"' if line.startswith("RUN_MODE = ") else line
                                             for line in cell.source.splitlines())
        client = NotebookClient(notebook, timeout=None, kernel_name="pricecatcher",
                                resources={"metadata": {"path": str(REPOSITORY)}})
        # Each notebook starts with a fresh kernel. A failed cell stops the run.
        try:
            client.execute()
        finally:
            # Preserve completed cells and the error for inspection if execution fails.
            output_folder.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, output_folder / filename)

    if prepare_needed:
        if (not cfg.STAGE1_COMBINED_PARQUET.exists() or not (cfg.STAGE1_DIR / "source_manifest.json").exists()
                or not (cfg.STAGE1_DIR / "executed" / "01_integrate_pricecatcher.ipynb").exists()):
            run_notebook("01_integrate_pricecatcher.ipynb", cfg.STAGE1_DIR / "executed")
        for filename in ["02_data_understanding_and_quality.ipynb", "03_cleaning_and_outliers.ipynb",
                         "04_weekly_aggregation.ipynb", "05_features_and_labels.ipynb"]:
            run_notebook(filename, cfg.RUN_DIR / "executed")
    if args.model_mode:
        original_dir, reproduction_dir = cfg.MODEL_DIRS
        model_dir = original_dir
        if model_only and not any((original_dir / name).exists() for name in ["best_model.joblib", "model_manifest.json"]):
            if args.model_mode == "auto" or any((reproduction_dir / name).exists() for name in ["best_model.joblib", "model_manifest.json"]):
                model_dir = reproduction_dir
        run_notebook("06_model_training_and_evaluation.ipynb", model_dir / "supplementary" / "executed")
    print(f"Completed. Outputs: {model_dir if args.model_mode else cfg.RUN_DIR}")


if __name__ == "__main__":
    main()
