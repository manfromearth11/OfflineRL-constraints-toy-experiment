"""Lambda sweep for regularized methods in multimodal offline RL.

Runs:
  - OT/L2 methods with explicit lambda sweep
  - BC / Forward KL / Reverse KL baselines (single value per seed)

Outputs:
  - results/lambda_sweep_raw.csv
  - results/lambda_sweep_summary.csv
  - results/lambda_sweep_curve.png
  - results/lambda_sweep_heatmap.png
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.bandit_env import generate_dataset, ground_truth_q, noisy_q, optimal_action
from src.methods.bc import BehavioralCloning
from src.methods.kl_forward import ForwardKLPolicy
from src.methods.kl_reverse import ReverseKLPolicy
from src.methods.l2_constraint import L2ConstraintPolicy
from src.methods.ppl import PartialOTPolicy
from src.methods.ot_unbalanced import UnbalancedOTPolicy
from src.methods.ot_wasserstein import WassersteinPolicy
from src.models.behavior_model import BehaviorModel, FittedBehaviorModel
from src.models.policy_net import GMMPolicy, GaussianPolicy
from src.training.trainer import Trainer
from src.visualization.plots import make_state_grid


METHOD_LABELS = {
    "bc": "BC",
    "forward_kl": "Forward KL",
    "reverse_kl": "Reverse KL",
    "wasserstein": "Wasserstein",
    "partial_ot": "Partial OT",
    "unbalanced_ot": "Unbalanced OT",
    "l2_constraint": "L2 Constraint",
}

OT_METHODS = {"wasserstein", "partial_ot", "unbalanced_ot", "l2_constraint"}
BASELINE_METHODS = {"bc", "forward_kl", "reverse_kl"}


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_policy(config: dict):
    policy_cfg = config.get("policy", {})
    if policy_cfg.get("type") == "gmm":
        return GMMPolicy(n_components=policy_cfg.get("n_components", 5))
    return GaussianPolicy()


def build_method(
    method_key: str,
    base_config: dict,
    env_config: dict,
    behavior_model,
    q_fn,
    lam_value: float | None,
    ot_n_samples: int,
    ot_n_states: int,
):
    if method_key == "bc":
        return BehavioralCloning(make_policy(base_config), copy.deepcopy(base_config.get("bc", {})))

    if method_key == "forward_kl":
        cfg = copy.deepcopy(base_config.get("kl_forward", {}))
        return ForwardKLPolicy(make_policy(base_config), cfg, q_fn=q_fn)

    if method_key == "reverse_kl":
        cfg = copy.deepcopy(base_config.get("kl_reverse", {}))
        return ReverseKLPolicy(make_policy(base_config), cfg, behavior_model, q_fn=q_fn)

    if method_key == "wasserstein":
        cfg = copy.deepcopy(base_config.get("ot_wasserstein", {}))
        cfg["lam"] = float(lam_value)
        cfg["n_ot_samples"] = ot_n_samples
        cfg["n_ot_states"] = ot_n_states
        return WassersteinPolicy(make_policy(base_config), cfg, env_config, q_fn=q_fn)

    if method_key == "partial_ot":
        cfg = copy.deepcopy(base_config.get("ot_partial", {}))
        cfg["lam"] = float(lam_value)
        cfg["n_ot_samples"] = ot_n_samples
        cfg["n_ot_states"] = ot_n_states
        return PartialOTPolicy(make_policy(base_config), cfg, env_config, q_fn=q_fn)

    if method_key == "unbalanced_ot":
        cfg = copy.deepcopy(base_config.get("ot_unbalanced", {}))
        cfg["lam"] = float(lam_value)
        cfg["n_ot_samples"] = ot_n_samples
        cfg["n_ot_states"] = ot_n_states
        return UnbalancedOTPolicy(make_policy(base_config), cfg, env_config, q_fn=q_fn)

    if method_key == "l2_constraint":
        cfg = copy.deepcopy(base_config.get("l2_constraint", {}))
        cfg["lam"] = float(lam_value)
        return L2ConstraintPolicy(make_policy(base_config), cfg, q_fn=q_fn)

    raise ValueError(f"Unknown method key: {method_key}")


def evaluate_mean_reward(trainer: Trainer, eval_grid_n: int, eval_samples: int) -> float:
    grid, _, _ = make_state_grid(n=eval_grid_n)
    s_grid = torch.tensor(grid, dtype=torch.float32)
    r = trainer.evaluate_reward(s_grid, n_samples=eval_samples)
    return float(r.mean().item())


def parse_xy(text: str) -> Tuple[float, float]:
    x, y = text.split(",")
    return float(x.strip()), float(y.strip())


@torch.no_grad()
def evaluate_good_mode_recall(
    policy,
    env_config: dict,
    state_xy: Tuple[float, float],
    n_samples: int,
    good_quantile: float,
    radius_mult: float,
    min_hits: int,
) -> float:
    """Recall of nearest (good) modes at one representative state."""
    modes = env_config.get("modes", [])
    if not modes:
        return float("nan")

    s = np.array(state_xy, dtype=np.float32)
    s_tile = np.tile(s, (n_samples, 1))
    a_star = optimal_action(s)

    mode_info = []
    for m in modes:
        offset = np.array(m["offset"], dtype=np.float32)
        mode_info.append(
            {
                "offset": offset,
                "std": float(m["std"]),
                "dist": float(np.linalg.norm(offset)),
            }
        )
    mode_info.sort(key=lambda x: x["dist"])
    n_good = max(1, int(math.ceil(len(mode_info) * good_quantile)))
    good_modes = mode_info[:n_good]

    a = policy.sample(torch.tensor(s_tile, dtype=torch.float32)).detach().cpu().numpy()
    recalled = 0
    for m in good_modes:
        center = a_star + m["offset"]
        radius = radius_mult * m["std"]
        d = np.linalg.norm(a - center[None, :], axis=1)
        if int((d <= radius).sum()) >= min_hits:
            recalled += 1
    return recalled / len(good_modes)


def aggregate_results(records: List[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, float], List[float]] = {}
    recall_grouped: Dict[Tuple[str, float], List[float]] = {}
    for rec in records:
        key = (rec["method"], rec["lambda"])
        grouped.setdefault(key, []).append(rec["mean_reward"])
        if "good_mode_recall" in rec and rec["good_mode_recall"] == rec["good_mode_recall"]:
            recall_grouped.setdefault(key, []).append(rec["good_mode_recall"])

    summary = []
    for (method, lam), values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        stderr = float(std / math.sqrt(len(arr)))
        summary.append(
            {
                "method": method,
                "lambda": lam,
                "n_seeds": len(arr),
                "mean_reward_mean": mean,
                "mean_reward_std": std,
                "mean_reward_stderr": stderr,
            }
        )
        if (method, lam) in recall_grouped and recall_grouped[(method, lam)]:
            r_arr = np.asarray(recall_grouped[(method, lam)], dtype=np.float64)
            summary[-1]["good_mode_recall_mean"] = float(r_arr.mean())
            summary[-1]["good_mode_recall_std"] = float(r_arr.std(ddof=0))
            summary[-1]["good_mode_recall_stderr"] = float(r_arr.std(ddof=0) / math.sqrt(len(r_arr)))
    summary.sort(key=lambda x: (x["method"], x["lambda"]))
    return summary


def save_csv(path: str, rows: List[dict], fieldnames: List[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved: {path}")


def plot_lambda_curve(
    summary: List[dict],
    ot_methods: List[str],
    baseline_methods: List[str],
    lambdas: List[float],
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in ot_methods:
        rows = [r for r in summary if r["method"] == method]
        rows.sort(key=lambda x: x["lambda"])
        x = np.array([r["lambda"] for r in rows], dtype=np.float64)
        y = np.array([r["mean_reward_mean"] for r in rows], dtype=np.float64)
        e = np.array([r["mean_reward_stderr"] for r in rows], dtype=np.float64)
        ax.errorbar(
            x,
            y,
            yerr=e,
            marker="o",
            linewidth=2.0,
            capsize=3.0,
            label=METHOD_LABELS.get(method, method),
        )

    line_styles = {
        "bc": ("--", "#666666"),
        "forward_kl": (":", "#333333"),
        "reverse_kl": ("-.", "#111111"),
    }
    for method in baseline_methods:
        rows = [r for r in summary if r["method"] == method]
        if not rows:
            continue
        y = rows[0]["mean_reward_mean"]
        e = rows[0]["mean_reward_stderr"]
        style, color = line_styles.get(method, ("--", "black"))
        ax.axhline(y=y, linestyle=style, color=color, linewidth=1.8, label=f"{METHOD_LABELS[method]} baseline")
        ax.fill_between([min(lambdas), max(lambdas)], [y - e, y - e], [y + e, y + e], color=color, alpha=0.06)

    ax.set_xscale("log", base=10)
    ax.set_xlabel("Lambda (OT regularization)")
    ax.set_ylabel("Mean reward (higher is better)")
    ax.set_title("OT lambda sweep with BC/KL baselines")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_lambda_heatmap(summary: List[dict], ot_methods: List[str], lambdas: List[float], save_path: str):
    heat = np.full((len(ot_methods), len(lambdas)), np.nan, dtype=np.float64)
    for i, m in enumerate(ot_methods):
        for j, lam in enumerate(lambdas):
            row = next(
                (x for x in summary if x["method"] == m and abs(x["lambda"] - lam) < 1e-12),
                None,
            )
            if row is not None:
                heat[i, j] = row["mean_reward_mean"]

    fig, ax = plt.subplots(figsize=(1.8 * len(lambdas), 1.4 * len(ot_methods) + 2))
    im = ax.imshow(heat, cmap="RdYlGn", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean reward")

    ax.set_xticks(np.arange(len(lambdas)))
    ax.set_xticklabels([f"{x:g}" for x in lambdas])
    ax.set_yticks(np.arange(len(ot_methods)))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in ot_methods])
    ax.set_xlabel("Lambda")
    ax.set_title("OT methods: lambda vs reward")

    for i in range(len(ot_methods)):
        for j in range(len(lambdas)):
            val = heat[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep regularization lambda in multimodal offline RL."
    )
    parser.add_argument("--config", type=str, default="configs/multimodal.yaml")
    parser.add_argument("--lambdas", type=str, default="0.1,0.3,1,3,10")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument(
        "--ot-methods",
        type=str,
        default="wasserstein,partial_ot,unbalanced_ot,l2_constraint",
    )
    parser.add_argument("--baseline-methods", type=str, default="bc,forward_kl,reverse_kl")
    parser.add_argument("--n-data", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--q-noise-std", type=float, default=3.0)
    parser.add_argument("--eval-grid", type=int, default=25)
    parser.add_argument("--eval-samples", type=int, default=40)
    parser.add_argument("--ot-samples", type=int, default=10)
    parser.add_argument("--ot-states", type=int, default=6)
    parser.add_argument("--compute-recall", action="store_true")
    parser.add_argument("--recall-state", type=str, default="0,0")
    parser.add_argument("--recall-samples", type=int, default=800)
    parser.add_argument("--recall-good-quantile", type=float, default=0.3)
    parser.add_argument("--recall-radius-mult", type=float, default=2.0)
    parser.add_argument("--recall-min-hits", type=int, default=5)
    parser.add_argument("--output-prefix", type=str, default="lambda_sweep")
    args = parser.parse_args()

    base_config = load_config(args.config)
    env_config = copy.deepcopy(base_config["env"])
    env_config["n_data"] = args.n_data

    train_config = copy.deepcopy(base_config.get("training", {}))
    train_config["n_epochs"] = args.epochs
    train_config["batch_size"] = args.batch_size
    train_config["lr"] = args.lr

    lambdas = sorted(parse_float_list(args.lambdas))
    seeds = parse_int_list(args.seeds)
    ot_methods = parse_str_list(args.ot_methods)
    baseline_methods = parse_str_list(args.baseline_methods)
    recall_state_xy = parse_xy(args.recall_state)

    os.makedirs("results", exist_ok=True)

    print("=" * 72)
    print("LAMBDA SWEEP (OT) WITH BC/KL BASELINES")
    print(f"OT methods: {ot_methods}")
    print(f"Baselines: {baseline_methods}")
    print(f"Lambdas: {lambdas}")
    print(f"Seeds: {seeds}")
    print(f"N={args.n_data}, epochs={args.epochs}, batch={args.batch_size}")
    print(f"Q noise std: {args.q_noise_std}")
    print(f"Compute recall: {args.compute_recall}")
    print("=" * 72)

    for m in ot_methods:
        if m not in OT_METHODS:
            raise ValueError(f"Invalid OT method: {m}")
    for m in baseline_methods:
        if m not in BASELINE_METHODS:
            raise ValueError(f"Invalid baseline method: {m}")

    raw_records: List[dict] = []

    for seed in seeds:
        print("\n" + "-" * 72)
        print(f"Seed {seed}")
        print("-" * 72)

        torch.manual_seed(seed)
        np.random.seed(seed)
        dataset = generate_dataset(env_config, seed=seed)

        corruption = base_config.get("corruption", {})
        if corruption.get("use_fitted_behavior", False):
            behavior_model = FittedBehaviorModel(dataset, n_grid=5, n_components=5)
        else:
            behavior_model = BehaviorModel(env_config)

        q_fn = partial(noisy_q, noise_std=args.q_noise_std) if args.q_noise_std > 0 else ground_truth_q

        # Baselines (single train per seed)
        for m_idx, method_key in enumerate(baseline_methods):
            run_seed = seed * 1000 + 31 * (m_idx + 1)
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)

            method = build_method(
                method_key=method_key,
                base_config=base_config,
                env_config=env_config,
                behavior_model=behavior_model,
                q_fn=q_fn,
                lam_value=None,
                ot_n_samples=args.ot_samples,
                ot_n_states=args.ot_states,
            )
            trainer = Trainer(method, dataset, train_config)
            losses = trainer.train(n_epochs=args.epochs)
            mean_reward = evaluate_mean_reward(trainer, args.eval_grid, args.eval_samples)
            if args.compute_recall:
                good_mode_recall = evaluate_good_mode_recall(
                    method.policy,
                    env_config,
                    state_xy=recall_state_xy,
                    n_samples=args.recall_samples,
                    good_quantile=args.recall_good_quantile,
                    radius_mult=args.recall_radius_mult,
                    min_hits=args.recall_min_hits,
                )
            else:
                good_mode_recall = float("nan")

            raw_records.append(
                {
                    "seed": seed,
                    "method": method_key,
                    "method_label": METHOD_LABELS[method_key],
                    "lambda": 1.0,
                    "mean_reward": mean_reward,
                    "good_mode_recall": good_mode_recall,
                    "final_loss": float(losses[-1]),
                    "is_baseline": 1,
                }
            )
            if args.compute_recall:
                print(
                    f"  {METHOD_LABELS[method_key]:<14} reward={mean_reward:8.4f} "
                    f"recall={good_mode_recall:6.3f} final_loss={losses[-1]:8.4f}"
                )
            else:
                print(f"  {METHOD_LABELS[method_key]:<14} reward={mean_reward:8.4f} final_loss={losses[-1]:8.4f}")

        # OT lambda sweep
        for lam in lambdas:
            print(f"\n[lambda={lam:g}]")
            for m_idx, method_key in enumerate(ot_methods):
                run_seed = seed * 10000 + int(round(lam * 1000)) + 97 * (m_idx + 1)
                torch.manual_seed(run_seed)
                np.random.seed(run_seed)

                method = build_method(
                    method_key=method_key,
                    base_config=base_config,
                    env_config=env_config,
                    behavior_model=behavior_model,
                    q_fn=q_fn,
                    lam_value=lam,
                    ot_n_samples=args.ot_samples,
                    ot_n_states=args.ot_states,
                )
                trainer = Trainer(method, dataset, train_config)
                losses = trainer.train(n_epochs=args.epochs)
                mean_reward = evaluate_mean_reward(trainer, args.eval_grid, args.eval_samples)
                if args.compute_recall:
                    good_mode_recall = evaluate_good_mode_recall(
                        method.policy,
                        env_config,
                        state_xy=recall_state_xy,
                        n_samples=args.recall_samples,
                        good_quantile=args.recall_good_quantile,
                        radius_mult=args.recall_radius_mult,
                        min_hits=args.recall_min_hits,
                    )
                else:
                    good_mode_recall = float("nan")

                raw_records.append(
                    {
                        "seed": seed,
                        "method": method_key,
                        "method_label": METHOD_LABELS[method_key],
                        "lambda": float(lam),
                        "mean_reward": mean_reward,
                        "good_mode_recall": good_mode_recall,
                        "final_loss": float(losses[-1]),
                        "is_baseline": 0,
                    }
                )
                if args.compute_recall:
                    print(
                        f"  {METHOD_LABELS[method_key]:<14} reward={mean_reward:8.4f} "
                        f"recall={good_mode_recall:6.3f} final_loss={losses[-1]:8.4f}"
                    )
                else:
                    print(f"  {METHOD_LABELS[method_key]:<14} reward={mean_reward:8.4f} final_loss={losses[-1]:8.4f}")

    summary = aggregate_results(raw_records)
    prefix = args.output_prefix
    raw_csv = f"results/{prefix}_raw.csv"
    summary_csv = f"results/{prefix}_summary.csv"
    curve_png = f"results/{prefix}_curve.png"
    heatmap_png = f"results/{prefix}_heatmap.png"

    raw_fieldnames = [
        "seed",
        "method",
        "method_label",
        "lambda",
        "mean_reward",
        "good_mode_recall",
        "final_loss",
        "is_baseline",
    ]
    summary_fieldnames = ["method", "lambda", "n_seeds", "mean_reward_mean", "mean_reward_std", "mean_reward_stderr"]
    if any("good_mode_recall_mean" in row for row in summary):
        summary_fieldnames += ["good_mode_recall_mean", "good_mode_recall_std", "good_mode_recall_stderr"]

    save_csv(
        raw_csv,
        raw_records,
        fieldnames=raw_fieldnames,
    )
    save_csv(
        summary_csv,
        summary,
        fieldnames=summary_fieldnames,
    )
    plot_lambda_curve(summary, ot_methods, baseline_methods, lambdas, curve_png)
    plot_lambda_heatmap(summary, ot_methods, lambdas, heatmap_png)

    print("\n" + "=" * 72)
    print("Summary (mean reward over seeds)")
    print("=" * 72)
    for method_key in baseline_methods + ot_methods:
        rows = [r for r in summary if r["method"] == method_key]
        rows.sort(key=lambda x: x["lambda"])
        print(f"\n{METHOD_LABELS[method_key]}")
        for row in rows:
            if "good_mode_recall_mean" in row:
                print(
                    f"  lambda={row['lambda']:>6g} "
                    f"mean={row['mean_reward_mean']:>8.4f} "
                    f"std={row['mean_reward_std']:>7.4f} "
                    f"recall={row['good_mode_recall_mean']:.3f}"
                )
            else:
                print(
                    f"  lambda={row['lambda']:>6g} "
                    f"mean={row['mean_reward_mean']:>8.4f} "
                    f"std={row['mean_reward_std']:>7.4f}"
                )

    print("\nDone.")
    print(f"Raw CSV: {raw_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Curve: {curve_png}")
    print(f"Heatmap: {heatmap_png}")


if __name__ == "__main__":
    main()
