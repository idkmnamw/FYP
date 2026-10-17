"""Run the same readable notebooks for one experiment or all expanding folds."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parent
EXPERIMENTS = ["fold_1", "fold_2", "fold_3", "holdout"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS + ["all"], default="holdout")
    parser.add_argument("--diagnostics", action="store_true", help="Also run optional Notebook 90")
    parser.add_argument("--refresh", action="store_true", help="Regenerate preparation for a study without a saved model")
    parser.add_argument("--model-mode", choices=["train", "reuse"], help="Also execute 06 after preparation; reuse never regenerates inputs")
    args = parser.parse_args()
    if args.refresh and args.model_mode == "reuse":
        parser.error("--refresh regenerates preparation; it cannot be combined with --model-mode reuse.")
    if args.experiment == "all":
        # Separate processes keep settings and notebook kernels isolated between folds.
        for experiment in ([] if args.model_mode == "reuse" else EXPERIMENTS):
            command = [sys.executable, str(Path(__file__).resolve()), "--experiment", experiment]
            if args.diagnostics:
                command.append("--diagnostics")
            if args.refresh:
                command.append("--refresh")
            subprocess.run(command, check=True, cwd=REPOSITORY)
        if args.model_mode is None:
            return
        args.experiment = "holdout"
        if args.model_mode == "train":
            args.diagnostics = False  # Each child experiment already ran its diagnostics.

    os.environ["PRICECATCHER_EXPERIMENT"] = args.experiment
    import pricecatcher_config as cfg

    # Frozen-model reuse verifies prepared inputs in 06. New preparation also
    # checks actual source bytes, even when the runner can reuse prepared files.
    if args.model_mode != "reuse" and not (cfg.PROJECT_ROOT / "06_models" / "best_model.joblib").exists():
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
    prepare_needed = (args.refresh or not complete) and args.model_mode != "reuse"
    if prepare_needed and (cfg.PROJECT_ROOT / "06_models" / "best_model.joblib").exists():
        raise SystemExit("A completed model exists. Restore its original prepared files instead of regenerating them. Use a separate PRICECATCHER_DATA_DIR for a new study.")
    if not prepare_needed and not args.diagnostics and args.model_mode is None:
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
                    cell.source = cell.source.replace('RUN_MODE = "reuse"', f'RUN_MODE = "{args.model_mode}"')
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
        if not cfg.STAGE1_COMBINED_PARQUET.exists() or not (cfg.STAGE1_DIR / "source_manifest.json").exists():
            run_notebook("01_integrate_pricecatcher.ipynb", cfg.STAGE1_DIR / "executed")
        for filename in ["02_data_understanding_and_quality.ipynb", "03_cleaning_and_outliers.ipynb",
                         "04_weekly_aggregation.ipynb", "05_features_and_labels.ipynb"]:
            run_notebook(filename, cfg.RUN_DIR / "executed")
    if args.diagnostics:
        run_notebook("90_optional_source_diagnostics.ipynb", cfg.RUN_DIR / "executed")
    if args.model_mode:
        run_notebook("06_model_training_and_evaluation.ipynb", cfg.PROJECT_ROOT / "06_models" / "supplementary" / "executed")
    print(f"Completed. Outputs: {cfg.RUN_DIR}")


if __name__ == "__main__":
    main()
