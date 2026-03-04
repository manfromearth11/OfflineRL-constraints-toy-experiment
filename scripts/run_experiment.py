#!/usr/bin/env python3
"""Single script entrypoint for all experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_3AXIS = PROJECT_ROOT / "src" / "runners" / "run_3axis_regime_sweep.py"
RUN_LAMBDA = PROJECT_ROOT / "src" / "runners" / "run_lambda_sweep_ot.py"


def run(cmd):
    print("=" * 88)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def run_smoke4(args, py):
    run(
        [
            py,
            str(RUN_3AXIS),
            "--base-config",
            args.base_config,
            "--dims",
            "2,4",
            "--seeds",
            "0",
            "--epochs",
            "2",
            "--n-data",
            "3000",
            "--scenarios",
            "baseline",
            "--modalities",
            "mid",
            "--q-levels",
            "mid",
            "--scenario-level",
            "moderate",
            "--output-prefix",
            f"{args.output_prefix}_smoke4",
            "--reset-results",
        ]
    )


def run_full4(args, py):
    run(
        [
            py,
            str(RUN_3AXIS),
            "--base-config",
            args.base_config,
            "--dims",
            args.dims,
            "--seeds",
            args.seeds,
            "--epochs",
            str(args.epochs),
            "--n-data",
            str(args.n_data),
            "--scenarios",
            args.scenarios,
            "--modalities",
            "low,mid,high",
            "--q-levels",
            "clean,mid,noisy",
            "--scenario-level",
            "moderate",
            "--output-prefix",
            args.output_prefix,
            "--reset-results",
        ]
    )


def run_lambda(args, py):
    run(
        [
            py,
            str(RUN_LAMBDA),
            "--config",
            args.lambda_config,
            "--lambdas",
            args.lambdas,
            "--ot-methods",
            args.ot_methods,
            "--output-prefix",
            args.lambda_output_prefix,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified experiment runner")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full", "all", "lambda", "smoke4", "full4"],
        default="smoke4",
    )

    parser.add_argument("--base-config", type=str, default="configs/ot_wins_dim_sweep.yaml")
    parser.add_argument("--dims", type=str, default="2,4,8")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-data", type=int, default=15000)
    parser.add_argument("--scenarios", type=str, default="baseline,good_shift,rotated,anchor_corrupt")
    parser.add_argument("--output-prefix", type=str, default="regime_4axis_scenario_balanced")

    parser.add_argument("--lambda-config", type=str, default="configs/multimodal.yaml")
    parser.add_argument("--lambdas", type=str, default="0.1,0.3,1,3,10")
    parser.add_argument("--ot-methods", type=str, default="wasserstein,partial_ot,unbalanced_ot,l2_constraint")
    parser.add_argument("--lambda-output-prefix", type=str, default="lambda_sweep_pot_l2")
    args = parser.parse_args()

    py = sys.executable

    if args.mode == "smoke":
        run_smoke4(args, py)
    elif args.mode == "full":
        run_full4(args, py)
    elif args.mode == "all":
        run_smoke4(args, py)
        run_full4(args, py)
    elif args.mode == "smoke4":
        run_smoke4(args, py)
    elif args.mode == "full4":
        run_full4(args, py)
    elif args.mode == "lambda":
        run_lambda(args, py)


if __name__ == "__main__":
    main()
