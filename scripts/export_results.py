"""Copy the small, final result tables from outputs/ into results/ for the repository.

Usage (from the repo root, after compare_hpo.py and final_eval.py):
    python scripts/export_results.py [--output-dir outputs] [--results results]

Writes results/hpo/<study>_trials.csv (one row per completed trial with the scored
objective after the failed-trial rule, the raw stored value and the parameters),
results/hpo/summary.csv, and the final test tables from outputs/final (model_metrics.csv,
method_summary.csv, patient_metrics.csv, results.md). Checkpoints, databases and
predictions stay in outputs/.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hpo import load_study_results

FINAL_FILES = ("model_metrics.csv", "method_summary.csv", "patient_metrics.csv", "results.md")


def export_trials(db_path: Path, out_path: Path) -> int:
    r = load_study_results(db_path)
    keys = list(r["params"][0]) if r["params"] else []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "val_error", "raw_value", "failed", *keys, "train_time_s"])
        for i, (value, raw, failed, params, seconds) in enumerate(
            zip(
                r["values"],
                r["raw_values"],
                r["failed"],
                r["params"],
                r["train_time_s"],
                strict=True,
            )
        ):
            writer.writerow([i, value, raw, failed, *(params[k] for k in keys), seconds])
    return len(r["values"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    outputs, results = Path(args.output_dir), Path(args.results)
    (results / "hpo").mkdir(parents=True, exist_ok=True)
    (results / "final").mkdir(parents=True, exist_ok=True)

    for db in sorted((outputs / "hpo").glob("*/study.db")):
        n = export_trials(db, results / "hpo" / f"{db.parent.name}_trials.csv")
        print(f"{db.parent.name}: {n} trials")
    shutil.copyfile(outputs / "hpo" / "comparison" / "summary.csv", results / "hpo" / "summary.csv")
    for name in FINAL_FILES:
        shutil.copyfile(outputs / "final" / name, results / "final" / name)
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
