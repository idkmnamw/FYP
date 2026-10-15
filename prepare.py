"""Run the same readable notebooks for one experiment or all expanding folds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
from nbclient import NotebookClient

REPOSITORY = Path(__file__).resolve().parent
EXPERIMENTS = ["fold_1", "fold_2", "fold_3", "holdout"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS + ["all"], default="holdout")
    parser.add_argument("--diagnostics", action="store_true", help="Also run optional Notebook 90")
    args = parser.parse_args()
    if args.experiment == "all":
        # Separate processes keep settings and notebook kernels isolated between folds.
        for experiment in EXPERIMENTS:
            command = [sys.executable, str(Path(__file__).resolve()), "--experiment", experiment]
            if args.diagnostics:
                command.append("--diagnostics")
            subprocess.run(command, check=True, cwd=REPOSITORY)
        return

    os.environ["PRICECATCHER_EXPERIMENT"] = args.experiment
    import pricecatcher_config as cfg

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
        print(f"{cfg.EXPERIMENT}: {filename}", flush=True)
        notebook = nbformat.read(REPOSITORY / filename, as_version=4)
        client = NotebookClient(notebook, timeout=None, kernel_name="pricecatcher",
                                resources={"metadata": {"path": str(REPOSITORY)}})
        # Each notebook starts with a fresh kernel. A failed cell stops the run.
        client.execute()
        output_folder.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output_folder / filename)

    if not cfg.STAGE1_COMBINED_PARQUET.exists() or not (cfg.STAGE1_DIR / "source_manifest.json").exists():
        run_notebook("01_integrate_pricecatcher.ipynb", cfg.STAGE1_DIR / "executed")
    for filename in ["02_data_understanding_and_quality.ipynb", "03_cleaning_and_outliers.ipynb",
                     "04_weekly_aggregation.ipynb", "05_features_and_labels.ipynb"]:
        run_notebook(filename, cfg.RUN_DIR / "executed")
    if args.diagnostics:
        run_notebook("90_optional_source_diagnostics.ipynb", cfg.RUN_DIR / "executed")
    print(f"Completed. Outputs: {cfg.RUN_DIR}")


if __name__ == "__main__":
    main()
