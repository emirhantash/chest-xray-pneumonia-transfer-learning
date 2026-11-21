"""Train AlexNet on the train subset and evaluate it on the validation subset.

Usage (from the repo root):
    python scripts/train.py --config configs/baseline.yaml [--data-root PATH]
        [--output-dir DIR] [--run-name NAME] [--resume] [--device cuda|mps|cpu]

Outputs go to <output_dir>/<run_name>/ (see src/runner.py). After an interruption, run
the same command with --resume to continue from the last completed epoch. The original
test set is not used.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runner import run_training
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data-root", help="Override data.root from the config")
    parser.add_argument("--output-dir", help="Override output_dir from the config")
    parser.add_argument("--run-name", help="Override run_name from the config")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the run from its last checkpoint (uses the config saved in it)",
    )
    parser.add_argument("--device", help="Force a device (default: CUDA, then MPS, then CPU)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.run_name:
        cfg["run_name"] = args.run_name
    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]

    summary = run_training(cfg, run_dir, resume=args.resume, device=args.device)
    val = summary["val_metrics"]
    print(
        f"\nFinal model (epoch {summary['epoch']}) on val: accuracy {val['accuracy']:.4f}, "
        f"error {val['error']:.4f}, recall {val['recall']:.4f}, "
        f"specificity {val['specificity']:.4f}, F1 {val['f1']:.4f}, AUC {val['auc']:.4f}"
    )
    print(f"Outputs: {run_dir}")


if __name__ == "__main__":
    main()
