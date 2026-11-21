"""Run (or resume) one hyperparameter-optimization study.

Usage (from the repo root):
    python scripts/hpo.py --method gp --seed 0 [--config configs/hpo.yaml]
        [--data-root PATH] [--output-dir DIR] [--backup-dir DIR] [--n-trials N]
        [--device cuda|mps|cpu]

Methods: gp (Gaussian-process Bayesian optimization) or random (Random Search).
The study is stored in <output_dir>/<method>_seed<seed>/study.db; rerunning the same
command continues until n_trials trials are complete. When done, best_config.yaml in the
study folder holds the best configuration for retraining with scripts/train.py.
The original test set is not used.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hpo import METHODS, run_study, study_name
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/hpo.yaml")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Sampler seed")
    parser.add_argument("--data-root", help="Override data.root of the base config")
    parser.add_argument("--output-dir", help="Override output_dir of the HPO config")
    parser.add_argument(
        "--backup-dir",
        help="Copy the study database here after every trial (e.g. a Google Drive folder)",
    )
    parser.add_argument("--n-trials", type=int, help="Override n_trials of the HPO config")
    parser.add_argument("--device", help="Force a device (default: CUDA, then MPS, then CPU)")
    args = parser.parse_args()

    hpo_cfg = load_config(args.config)
    base_cfg = load_config(hpo_cfg["base_config"])
    if args.data_root:
        base_cfg["data"]["root"] = args.data_root
    output_dir = Path(args.output_dir or hpo_cfg["output_dir"])
    name = study_name(args.method, args.seed)
    backup_dir = Path(args.backup_dir) / name if args.backup_dir else None

    study = run_study(
        hpo_cfg,
        base_cfg,
        args.method,
        args.seed,
        output_dir / name,
        n_trials=args.n_trials,
        device=args.device,
        backup_dir=backup_dir,
    )
    best = study.best_trial
    print(f"\nBest of {name}: trial {best.number}, val error {best.value:.6f}, {best.params}")
    print(f"Best config: {output_dir / name / 'best_config.yaml'}")


if __name__ == "__main__":
    main()
