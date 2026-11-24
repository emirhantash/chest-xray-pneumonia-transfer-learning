"""Final evaluation (Phase 4): the baseline and the retrained best HPO models on the test set.

Usage (from the repo root):
    python scripts/final_eval.py --final --data-root PATH [--runs outputs/baseline ...]
        [--out outputs/final] [--n-boot 2000]

This is the only step that uses the original test set. Each run is evaluated once with
``evaluate_run(split="test", final=True)``; a run that already has
<run>/eval_test/test_predictions.csv with log-odds is not evaluated again. Default runs:
outputs/baseline and every retrained outputs/best_<method>_seed<n>. AUC and ROC curves
use the log-odds, because confident probabilities saturate at 1.0 and tie.

Writes to --out: model_metrics.csv (metrics, 95 % patient-level bootstrap intervals,
McNemar test and accuracy difference vs the baseline), method_summary.csv (mean ± std
over seeds), results.md (both tables in Markdown), and roc_test.png.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import roc_points
from src.plots import METHOD_COLORS, plot_roc_overlay
from src.report import (
    markdown_tables,
    method_rows,
    model_rows,
    read_predictions,
    run_method,
    write_csv,
)
from src.runner import evaluate_run

LINESTYLES = ["-", "--", ":", "-."]


def _has_log_odds(pred_path: Path) -> bool:
    """Older prediction files lack the log-odds column; their rounded probabilities tie
    for confident predictions and give a wrong AUC, so such runs are evaluated again."""
    with open(pred_path, encoding="utf-8") as f:
        return "logit_pneumonia" in f.readline()


def default_runs(output_dir: Path) -> list[Path]:
    runs = [output_dir / "baseline"]
    for method in ("gp", "random"):
        runs += sorted(output_dir.glob(f"best_{method}_seed*"))
    return [r for r in runs if r.is_dir()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final", action="store_true", help="Confirm the final evaluation")
    parser.add_argument("--runs", nargs="+", help="Run folders (default: baseline + best_*)")
    parser.add_argument("--output-dir", default="outputs", help="Where the runs live")
    parser.add_argument("--data-root", help="Override data.root saved in the runs")
    parser.add_argument("--out", default="outputs/final")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed")
    parser.add_argument("--device", help="Force a device (default: CUDA, then MPS, then CPU)")
    args = parser.parse_args()
    if not args.final:
        parser.error("This evaluates the test set (final evaluation); add --final to confirm.")

    runs = [Path(r) for r in args.runs] if args.runs else default_runs(Path(args.output_dir))
    if not runs:
        sys.exit("No runs found.")
    preds, val_errors, thresholds = {}, {}, set()
    for run in runs:
        pred_path = run / "eval_test" / "test_predictions.csv"
        if pred_path.is_file() and _has_log_odds(pred_path):
            print(f"{run.name}: reusing {pred_path}")
        else:
            print(f"{run.name}: evaluating on the test set")
            evaluate_run(
                run, split="test", final=True, data_root=args.data_root, device=args.device
            )
        report = json.loads((run / "eval_test" / "metrics.json").read_text(encoding="utf-8"))
        thresholds.add(report["threshold"])
        val = json.loads((run / "val_metrics.json").read_text(encoding="utf-8"))
        preds[run.name] = read_predictions(pred_path)
        val_errors[run.name] = val["metrics"]["error"]
    if len(thresholds) != 1:
        sys.exit(f"The runs use different thresholds: {sorted(thresholds)}")

    threshold = thresholds.pop()
    rows = model_rows(preds, val_errors, threshold, args.n_boot, args.seed)
    methods = method_rows(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out / "model_metrics.csv")
    write_csv(methods, out / "method_summary.csv")
    tables = markdown_tables(rows, methods)
    (out / "results.md").write_text(tables, encoding="utf-8")

    curves = []
    for row in rows:
        p = preds[row["model"]]
        fpr, tpr = roc_points(p["label"], p["score"])
        method, seed = run_method(row["model"])
        curves.append(
            {
                "label": f"{row['model']} (AUC {row['auc']:.4f})",
                "fpr": fpr,
                "tpr": tpr,
                "color": METHOD_COLORS.get(method, "black"),
                "linestyle": LINESTYLES[(seed or 0) % len(LINESTYLES)],
                "point": (1 - row["specificity"], row["recall"]),
                "point_label": f"operating point (threshold {threshold})",
            }
        )
    plot_roc_overlay(curves, out / "roc_test.png", "ROC on the test set")
    print("\n" + tables)
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
