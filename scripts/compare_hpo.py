"""Compare the HPO studies (Bayesian optimization vs Random Search) on validation error.

Usage (from the repo root):
    python scripts/compare_hpo.py [--hpo-dir outputs/hpo] [--config configs/hpo.yaml]
        [--baseline outputs/baseline/val_metrics.json] [--out outputs/hpo/comparison]

Reads every <hpo-dir>/*/study.db and writes summary.csv, best_so_far.png,
trial_errors.png, and params_vs_error.png. Studies may have different numbers of
completed trials; the mean curves use the common length.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hpo import STUDY_DB, load_study_results, summarize
from src.plots import plot_best_so_far, plot_params_vs_error, plot_trial_errors
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/hpo.yaml")
    parser.add_argument("--hpo-dir", help="Folder with one subfolder per study")
    parser.add_argument(
        "--baseline",
        default="outputs/baseline/val_metrics.json",
        help="val_metrics.json of the baseline run (reference line); skipped if missing",
    )
    parser.add_argument("--out", help="Output folder (default: <hpo-dir>/comparison)")
    args = parser.parse_args()

    hpo_cfg = load_config(args.config)
    hpo_dir = Path(args.hpo_dir or hpo_cfg["output_dir"])
    out_dir = Path(args.out) if args.out else hpo_dir / "comparison"
    results = [load_study_results(p) for p in sorted(hpo_dir.glob(f"*/{STUDY_DB}"))]
    results = [r for r in results if r["values"]]
    if not results:
        sys.exit(f"No study with completed trials under {hpo_dir}")

    baseline_error = None
    baseline_path = Path(args.baseline)
    if baseline_path.is_file():
        baseline_error = json.loads(baseline_path.read_text(encoding="utf-8"))["metrics"]["error"]

    rows = summarize(results, baseline_error)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    plot_best_so_far(results, baseline_error, out_dir / "best_so_far.png")
    plot_trial_errors(results, baseline_error, out_dir / "trial_errors.png")
    plot_params_vs_error(results, hpo_cfg["search_space"], out_dir / "params_vs_error.png")

    if baseline_error is not None:
        print(f"Baseline val error: {baseline_error:.4f}")
    print(f"{'study':16s} {'trials':>6s} {'best':>8s} {'median':>8s} {'beat@':>6s} {'hours':>6s}")
    for row in rows:
        beat = row.get("trials_to_beat_baseline")
        print(
            f"{row['study']:16s} {row['n_trials']:6d} {row['best_error']:8.4f} "
            f"{row['median_error']:8.4f} {beat if beat is not None else '-':>6} "
            f"{row['total_time_h']:6.2f}"
        )
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
