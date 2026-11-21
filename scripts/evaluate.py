"""Evaluate the final model of a completed training run.

Usage (from the repo root):
    python scripts/evaluate.py --run outputs/baseline [--data-root PATH]      # validation
    python scripts/evaluate.py --run outputs/baseline --split test --final    # final test

The original test set is reserved for the final evaluation of the selected models: it
is only evaluated with an explicit --split test --final.
Outputs go to <run>/eval_<split>/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runner import evaluate_run


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="Run folder, e.g. outputs/baseline")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--final",
        action="store_true",
        help="Confirm the final evaluation; required for --split test",
    )
    parser.add_argument("--data-root", help="Override data.root saved in the run")
    parser.add_argument("--device", help="Force a device (default: CUDA, then MPS, then CPU)")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("--split test is the final evaluation; add --final to confirm.")

    metrics = evaluate_run(
        args.run, split=args.split, final=args.final, data_root=args.data_root, device=args.device
    )
    print(f"{args.split}: " + ", ".join(f"{k} {v:.4g}" for k, v in metrics.items()))
    print(f"Outputs: {Path(args.run) / f'eval_{args.split}'}")


if __name__ == "__main__":
    main()
