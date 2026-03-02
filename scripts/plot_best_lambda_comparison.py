"""Create a reward_free_comparison-style plot using OT best lambdas from summary CSV."""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List

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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_state_list(text: str) -> List[List[float]]:
    # format: "0,0;1,1;-1,2;2,-1"
    states = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(",")
        states.append([float(x), float(y)])
    return states


def make_policy(config: dict):
    policy_cfg = config.get("policy", {})
    if policy_cfg.get("type") == "gmm":
        return GMMPolicy(n_components=policy_cfg.get("n_components", 5))
    return GaussianPolicy()


def get_best_lambdas(summary_csv: str) -> Dict[str, float]:
    target = {"wasserstein", "partial_ot", "unbalanced_ot"}
    grouped = {k: [] for k in target}
    with open(summary_csv) as f:
        rd = csv.DictReader(f)
        for r in rd:
            method = r["method"]
            if method in grouped:
                grouped[method].append(
                    {"lambda": float(r["lambda"]), "mean": float(r["mean_reward_mean"])}
                )
    best = {}
    for method, arr in grouped.items():
        if not arr:
            raise ValueError(f"No rows found for method '{method}' in {summary_csv}")
        row = max(arr, key=lambda x: x["mean"])
        best[method] = row["lambda"]
    return best


def plot_reward_style_comparison(results, representative_states, env_config, n_samples=500, save_path=None):
    """Same visual style as reward_free_comparison."""
    states = np.array(representative_states)
    method_names = list(results.keys())
    n_methods = len(method_names)
    n_states = len(states)

    fig, axes = plt.subplots(n_methods, n_states, figsize=(5 * n_states, 4.5 * n_methods))
    if n_methods == 1:
        axes = axes[np.newaxis, :]

    for j, s in enumerate(states):
        s_tile = np.tile(s, (n_samples, 1))
        a_opt = optimal_action(s)

        rng = np.random.default_rng(j + 42)
        a_behav = behavior_policy_sample(s_tile, env_config, rng)

        for i, name in enumerate(method_names):
            ax = axes[i, j]
            policy = results[name]["policy"]

            with torch.no_grad():
                s_t = torch.tensor(s_tile, dtype=torch.float32)
                a_policy = policy.sample(s_t).numpy()

            dist_from_opt = np.linalg.norm(a_behav - a_opt, axis=-1)
            ax.scatter(
                a_behav[:, 0],
                a_behav[:, 1],
                c=dist_from_opt,
                cmap="RdYlGn_r",
                alpha=0.15,
                s=5,
                vmin=0,
                vmax=5,
            )
            ax.scatter(a_policy[:, 0], a_policy[:, 1], c="C0", alpha=0.3, s=8, label="Policy")
            ax.scatter([a_opt[0]], [a_opt[1]], c="red", marker="*", s=300, zorder=10, label="Optimal")

            mu = (
                policy.mean_action(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
                .detach()
                .numpy()[0]
            )
            ax.scatter([mu[0]], [mu[1]], c="blue", marker="x", s=150, zorder=10)

            if i == 0:
                ax.set_title(f"s=({s[0]:.0f},{s[1]:.0f})", fontsize=13)
            if j == 0:
                ax.set_ylabel(name, fontsize=12, fontweight="bold")
            ax.set_xlim(a_opt[0] - 5, a_opt[0] + 6)
            ax.set_ylim(a_opt[1] - 5, a_opt[1] + 6)
            ax.set_aspect("equal")
            if i == 0 and j == n_states - 1:
                ax.legend(fontsize=7, loc="upper right")

    plt.suptitle(
        "Best-Lambda Comparison: Behavior (dist-colored) + Policy",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot reward_free-style comparison with OT best lambdas.")
    parser.add_argument("--config", type=str, default="configs/low_snr.yaml")
    parser.add_argument("--summary-csv", type=str, required=True)
    parser.add_argument("--q-noise-std", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-data", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ot-samples", type=int, default=10)
    parser.add_argument("--ot-states", type=int, default=6)
    parser.add_argument("--n-plot-samples", type=int, default=600)
    parser.add_argument("--states", type=str, default="")
    parser.add_argument(
        "--output",
        type=str,
        default="results/lambda_best_reward_style_comparison.png",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    env_config = copy.deepcopy(config["env"])
    env_config["n_data"] = args.n_data

    train_config = copy.deepcopy(config.get("training", {}))
    train_config["n_epochs"] = args.epochs
    train_config["batch_size"] = args.batch_size
    train_config["lr"] = args.lr
    train_config["seed"] = args.seed

    os.makedirs("results", exist_ok=True)

    best_lambdas = get_best_lambdas(args.summary_csv)
    print("Best lambdas from summary:")
    for k, v in best_lambdas.items():
        print(f"  {k}: {v:g}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = generate_dataset(env_config, seed=args.seed)

    corruption = config.get("corruption", {})
    if corruption.get("use_fitted_behavior", False):
        behavior_model = FittedBehaviorModel(dataset, n_grid=5, n_components=5)
    else:
        behavior_model = BehaviorModel(env_config)

    q_fn = partial(noisy_q, noise_std=args.q_noise_std) if args.q_noise_std > 0 else ground_truth_q

    methods = {}
    methods["BC"] = BehavioralCloning(make_policy(config), config.get("bc", {}))
    methods["Forward KL"] = ForwardKLPolicy(make_policy(config), config.get("kl_forward", {}), q_fn=q_fn)
    methods["Reverse KL"] = ReverseKLPolicy(
        make_policy(config), config.get("kl_reverse", {}), behavior_model, q_fn=q_fn
    )

    w_cfg = copy.deepcopy(config.get("ot_wasserstein", {}))
    w_cfg["lam"] = best_lambdas["wasserstein"]
    w_cfg["n_ot_samples"] = args.ot_samples
    w_cfg["n_ot_states"] = args.ot_states
    methods[f"Wasserstein (λ={best_lambdas['wasserstein']:g})"] = WassersteinPolicy(
        make_policy(config), w_cfg, env_config, q_fn=q_fn
    )

    p_cfg = copy.deepcopy(config.get("ot_partial", {}))
    p_cfg["lam"] = best_lambdas["partial_ot"]
    p_cfg["n_ot_samples"] = args.ot_samples
    p_cfg["n_ot_states"] = args.ot_states
    methods[f"Partial OT (λ={best_lambdas['partial_ot']:g})"] = PartialOTPolicy(
        make_policy(config), p_cfg, env_config, q_fn=q_fn
    )

    u_cfg = copy.deepcopy(config.get("ot_unbalanced", {}))
    u_cfg["lam"] = best_lambdas["unbalanced_ot"]
    u_cfg["n_ot_samples"] = args.ot_samples
    u_cfg["n_ot_states"] = args.ot_states
    methods[f"Unbalanced OT (λ={best_lambdas['unbalanced_ot']:g})"] = UnbalancedOTPolicy(
        make_policy(config), u_cfg, env_config, q_fn=q_fn
    )

    results = {}
    for name, method in methods.items():
        print(f"\nTraining {name} ...")
        trainer = Trainer(method, dataset, train_config)
        losses = trainer.train(n_epochs=args.epochs)
        results[name] = {"policy": method.policy, "losses": losses}
        print(f"  Final loss: {losses[-1]:.4f}")

    if args.states.strip():
        rep_states = parse_state_list(args.states)
    else:
        rep_states = config.get("viz", {}).get(
            "representative_states", [[0, 0], [1, 1], [-1, 2], [2, -1]]
        )

    plot_reward_style_comparison(
        results,
        rep_states,
        env_config,
        n_samples=args.n_plot_samples,
        save_path=args.output,
    )

    print("\nDone.")
    print(f"Saved comparison plot: {args.output}")


if __name__ == "__main__":
    main()
