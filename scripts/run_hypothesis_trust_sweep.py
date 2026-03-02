"""Hypothesis test: when does KL (Q-trust) vs OT (BC-trust) work better?

This script runs a directional sweep over trust ratio:
    ratio > 1.0  -> trust Q more
    ratio < 1.0  -> trust behavior distribution (BC prior) more

Mapping to each method:
    KL methods: alpha_eff = alpha_base / ratio
    OT methods: lam_eff   = lam_base / ratio

Outputs:
    - results/hypothesis_trust_sweep_raw.csv
    - results/hypothesis_trust_sweep_summary.csv
    - results/hypothesis_trust_curve.png
    - results/hypothesis_trust_heatmap.png
    - results/hypothesis_policy_snapshots.png
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.bandit_env import (
    behavior_policy_sample,
    generate_dataset,
    ground_truth_q,
    noisy_q,
    optimal_action,
)
from src.methods.bc import BehavioralCloning
from src.methods.kl_forward import ForwardKLPolicy
from src.methods.kl_reverse import ReverseKLPolicy
from src.methods.ot_partial import PartialOTPolicy
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
}


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
    ratio: float,
    ot_n_samples: int,
    ot_n_states: int,
):
    if method_key == "bc":
        return BehavioralCloning(make_policy(base_config), copy.deepcopy(base_config.get("bc", {})))

    if method_key == "forward_kl":
        cfg = copy.deepcopy(base_config.get("kl_forward", {}))
        base_alpha = float(cfg.get("alpha", 1.0))
        cfg["alpha"] = max(1e-4, base_alpha / ratio)
        return ForwardKLPolicy(make_policy(base_config), cfg, q_fn=q_fn)

    if method_key == "reverse_kl":
        cfg = copy.deepcopy(base_config.get("kl_reverse", {}))
        base_alpha = float(cfg.get("alpha", 1.0))
        cfg["alpha"] = max(1e-4, base_alpha / ratio)
        return ReverseKLPolicy(make_policy(base_config), cfg, behavior_model, q_fn=q_fn)

    if method_key == "wasserstein":
        cfg = copy.deepcopy(base_config.get("ot_wasserstein", {}))
        base_lam = float(cfg.get("lam", 1.0))
        cfg["lam"] = max(1e-4, base_lam / ratio)
        cfg["n_ot_samples"] = ot_n_samples
        cfg["n_ot_states"] = ot_n_states
        return WassersteinPolicy(make_policy(base_config), cfg, env_config, q_fn=q_fn)

    if method_key == "partial_ot":
        cfg = copy.deepcopy(base_config.get("ot_partial", {}))
        base_lam = float(cfg.get("lam", 1.0))
        cfg["lam"] = max(1e-4, base_lam / ratio)
        cfg["n_ot_samples"] = ot_n_samples
        cfg["n_ot_states"] = ot_n_states
        return PartialOTPolicy(make_policy(base_config), cfg, env_config, q_fn=q_fn)

    if method_key == "unbalanced_ot":
        cfg = copy.deepcopy(base_config.get("ot_unbalanced", {}))
        base_lam = float(cfg.get("lam", 1.0))
        cfg["lam"] = max(1e-4, base_lam / ratio)
        cfg["n_ot_samples"] = ot_n_samples
        cfg["n_ot_states"] = ot_n_states
        return UnbalancedOTPolicy(make_policy(base_config), cfg, env_config, q_fn=q_fn)

    raise ValueError(f"Unknown method key: {method_key}")


def evaluate_mean_reward(trainer: Trainer, eval_grid_n: int, eval_samples: int) -> float:
    grid, _, _ = make_state_grid(n=eval_grid_n)
    s_grid = torch.tensor(grid, dtype=torch.float32)
    r = trainer.evaluate_reward(s_grid, n_samples=eval_samples)
    return float(r.mean().item())


def aggregate_results(records: List[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, float], List[float]] = {}
    for rec in records:
        key = (rec["method"], rec["ratio"])
        grouped.setdefault(key, []).append(rec["mean_reward"])

    summary = []
    for (method, ratio), values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        stderr = float(std / math.sqrt(len(arr)))
        summary.append(
            {
                "method": method,
                "ratio": ratio,
                "n_seeds": len(arr),
                "mean_reward_mean": mean,
                "mean_reward_std": std,
                "mean_reward_stderr": stderr,
            }
        )
    summary.sort(key=lambda x: (x["method"], x["ratio"]))
    return summary


def save_csv(path: str, rows: List[dict], fieldnames: List[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved: {path}")


def plot_trust_curve(summary: List[dict], methods: List[str], ratios: List[float], save_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axvspan(min(ratios), 1.0, color="#f3f3f3", alpha=0.8)
    ax.axvspan(1.0, max(ratios), color="#eef6ff", alpha=0.6)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)

    for method in methods:
        rows = [r for r in summary if r["method"] == method]
        rows.sort(key=lambda x: x["ratio"])
        x = np.array([r["ratio"] for r in rows], dtype=np.float64)
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

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Trust Ratio (Q / BC prior)")
    ax.set_ylabel("Mean reward (higher is better)")
    ax.set_title("Q-trust vs BC-trust sweep (multimodal offline RL)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.text(min(ratios), ax.get_ylim()[1], "BC trust ↑", ha="left", va="bottom", fontsize=10)
    ax.text(max(ratios), ax.get_ylim()[1], "Q trust ↑", ha="right", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_trust_heatmap(summary: List[dict], methods: List[str], ratios: List[float], save_path: str):
    heat = np.full((len(methods), len(ratios)), np.nan, dtype=np.float64)
    for i, m in enumerate(methods):
        for j, r in enumerate(ratios):
            row = next(
                (x for x in summary if x["method"] == m and abs(x["ratio"] - r) < 1e-12),
                None,
            )
            if row is not None:
                heat[i, j] = row["mean_reward_mean"]

    fig, ax = plt.subplots(figsize=(1.8 * len(ratios), 1.2 * len(methods) + 2))
    im = ax.imshow(heat, cmap="RdYlGn", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean reward")

    ax.set_xticks(np.arange(len(ratios)))
    ax.set_xticklabels([f"{r:g}" for r in ratios])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax.set_xlabel("Trust ratio (Q/BC)")
    ax.set_title("Mean reward heatmap")

    for i in range(len(methods)):
        for j in range(len(ratios)):
            val = heat[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_policy_snapshots(
    policies: Dict[Tuple[str, float], torch.nn.Module],
    env_config: dict,
    methods: List[str],
    ratios: List[float],
    representative_state: Tuple[float, float],
    n_samples: int,
    save_path: str,
):
    if not methods or not ratios:
        return

    s = np.array(representative_state, dtype=np.float32)
    s_tile = np.tile(s, (n_samples, 1))
    a_behav = behavior_policy_sample(s_tile, env_config, rng=np.random.default_rng(7))
    a_opt = optimal_action(s)

    fig, axes = plt.subplots(len(methods), len(ratios), figsize=(4.5 * len(ratios), 4 * len(methods)))
    axes = np.atleast_2d(axes)

    for i, method in enumerate(methods):
        for j, ratio in enumerate(ratios):
            ax = axes[i, j]
            key = (method, ratio)
            if key not in policies:
                ax.set_visible(False)
                continue
            policy = policies[key]
            with torch.no_grad():
                a_policy = policy.sample(torch.tensor(s_tile, dtype=torch.float32)).numpy()

            ax.scatter(a_behav[:, 0], a_behav[:, 1], c="gray", alpha=0.15, s=6, label="Behavior")
            ax.scatter(a_policy[:, 0], a_policy[:, 1], c="C0", alpha=0.35, s=6, label="Policy")
            ax.scatter([a_opt[0]], [a_opt[1]], c="red", marker="*", s=180, zorder=10, label="Optimal")

            mu = (
                policy.mean_action(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
                .detach()
                .numpy()[0]
            )
            ax.scatter([mu[0]], [mu[1]], c="blue", marker="x", s=100, zorder=10, label="Policy mean")

            if i == 0:
                ax.set_title(f"ratio={ratio:g}", fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(METHOD_LABELS.get(method, method), fontsize=12, fontweight="bold")
            ax.set_aspect("equal")
            ax.set_xlim(a_opt[0] - 6, a_opt[0] + 6)
            ax.set_ylim(a_opt[1] - 6, a_opt[1] + 6)
            if i == 0 and j == len(ratios) - 1:
                ax.legend(fontsize=8, loc="upper right")

    plt.suptitle(
        f"Policy snapshots at state s=({representative_state[0]}, {representative_state[1]})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Sweep Q-vs-BC trust in multimodal offline RL.")
    parser.add_argument("--config", type=str, default="configs/multimodal.yaml")
    parser.add_argument("--ratios", type=str, default="0.25,0.5,1,2,4")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--methods", type=str, default="bc,forward_kl,partial_ot")
    parser.add_argument("--n-data", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--q-noise-std", type=float, default=None)
    parser.add_argument("--eval-grid", type=int, default=30)
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--ot-samples", type=int, default=16)
    parser.add_argument("--ot-states", type=int, default=8)
    parser.add_argument("--snapshot-methods", type=str, default="forward_kl,partial_ot")
    parser.add_argument("--snapshot-ratios", type=str, default="0.25,1,4")
    parser.add_argument("--snapshot-state", type=str, default="0,0")
    parser.add_argument("--snapshot-seed", type=int, default=0)
    args = parser.parse_args()

    base_config = load_config(args.config)
    env_config = copy.deepcopy(base_config["env"])
    env_config["n_data"] = args.n_data

    train_config = copy.deepcopy(base_config.get("training", {}))
    train_config["n_epochs"] = args.epochs
    train_config["batch_size"] = args.batch_size
    train_config["lr"] = args.lr

    corruption = copy.deepcopy(base_config.get("corruption", {}))
    if args.q_noise_std is not None:
        corruption["q_noise_std"] = args.q_noise_std

    ratios = sorted(parse_float_list(args.ratios))
    seeds = parse_int_list(args.seeds)
    methods = parse_str_list(args.methods)
    snapshot_methods = parse_str_list(args.snapshot_methods)
    snapshot_ratios = parse_float_list(args.snapshot_ratios)
    snapshot_state = tuple(float(x.strip()) for x in args.snapshot_state.split(","))

    os.makedirs("results", exist_ok=True)

    print("=" * 72)
    print("HYPOTHESIS SWEEP: Q-trust vs BC-trust")
    print(f"Methods: {methods}")
    print(f"Ratios (Q/BC): {ratios}")
    print(f"Seeds: {seeds}")
    print(f"N={args.n_data}, epochs={args.epochs}, batch={args.batch_size}")
    print(f"Q noise std: {corruption.get('q_noise_std', 0.0)}")
    print("=" * 72)

    raw_records: List[dict] = []
    snapshot_policies: Dict[Tuple[str, float], torch.nn.Module] = {}

    q_noise_std = corruption.get("q_noise_std", 0.0)

    for seed in seeds:
        print("\n" + "-" * 72)
        print(f"Seed {seed}")
        print("-" * 72)

        torch.manual_seed(seed)
        np.random.seed(seed)
        dataset = generate_dataset(env_config, seed=seed)

        if corruption.get("use_fitted_behavior", False):
            behavior_model = FittedBehaviorModel(dataset, n_grid=5, n_components=5)
        else:
            behavior_model = BehaviorModel(env_config)

        q_fn = partial(noisy_q, noise_std=q_noise_std) if q_noise_std > 0 else ground_truth_q

        for ratio in ratios:
            print(f"\n[ratio={ratio:g}]")
            for m_idx, method_key in enumerate(methods):
                if method_key not in METHOD_LABELS:
                    raise ValueError(f"Unknown method in --methods: {method_key}")

                run_seed = seed * 1000 + m_idx * 97 + int(round(ratio * 100))
                torch.manual_seed(run_seed)
                np.random.seed(run_seed)

                method = build_method(
                    method_key=method_key,
                    base_config=base_config,
                    env_config=env_config,
                    behavior_model=behavior_model,
                    q_fn=q_fn,
                    ratio=ratio,
                    ot_n_samples=args.ot_samples,
                    ot_n_states=args.ot_states,
                )
                trainer = Trainer(method, dataset, train_config)
                losses = trainer.train(n_epochs=args.epochs)
                mean_reward = evaluate_mean_reward(
                    trainer=trainer,
                    eval_grid_n=args.eval_grid,
                    eval_samples=args.eval_samples,
                )

                rec = {
                    "seed": seed,
                    "ratio": ratio,
                    "method": method_key,
                    "method_label": METHOD_LABELS.get(method_key, method_key),
                    "mean_reward": mean_reward,
                    "final_loss": float(losses[-1]),
                }
                raw_records.append(rec)
                print(
                    f"  {METHOD_LABELS[method_key]:<14} "
                    f"reward={mean_reward:8.4f}  final_loss={losses[-1]:8.4f}"
                )

                if (
                    seed == args.snapshot_seed
                    and method_key in snapshot_methods
                    and any(abs(ratio - rr) < 1e-12 for rr in snapshot_ratios)
                ):
                    snapshot_policies[(method_key, ratio)] = method.policy

    summary = aggregate_results(raw_records)

    raw_csv = "results/hypothesis_trust_sweep_raw.csv"
    summary_csv = "results/hypothesis_trust_sweep_summary.csv"
    curve_png = "results/hypothesis_trust_curve.png"
    heatmap_png = "results/hypothesis_trust_heatmap.png"
    snapshot_png = "results/hypothesis_policy_snapshots.png"

    save_csv(
        raw_csv,
        raw_records,
        fieldnames=["seed", "ratio", "method", "method_label", "mean_reward", "final_loss"],
    )
    save_csv(
        summary_csv,
        summary,
        fieldnames=[
            "method",
            "ratio",
            "n_seeds",
            "mean_reward_mean",
            "mean_reward_std",
            "mean_reward_stderr",
        ],
    )

    plot_trust_curve(summary, methods, ratios, curve_png)
    plot_trust_heatmap(summary, methods, ratios, heatmap_png)
    plot_policy_snapshots(
        policies=snapshot_policies,
        env_config=env_config,
        methods=snapshot_methods,
        ratios=snapshot_ratios,
        representative_state=snapshot_state,
        n_samples=500,
        save_path=snapshot_png,
    )

    print("\n" + "=" * 72)
    print("Summary (mean reward over seeds)")
    print("=" * 72)
    for method in methods:
        rows = [r for r in summary if r["method"] == method]
        rows.sort(key=lambda x: x["ratio"])
        print(f"\n{METHOD_LABELS.get(method, method)}")
        for row in rows:
            print(
                f"  ratio={row['ratio']:>5g} "
                f"mean={row['mean_reward_mean']:>8.4f} "
                f"std={row['mean_reward_std']:>7.4f}"
            )

    print("\nDone.")
    print(f"Raw CSV: {raw_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Curve: {curve_png}")
    print(f"Heatmap: {heatmap_png}")
    print(f"Snapshots: {snapshot_png}")


if __name__ == "__main__":
    main()
