"""Low SNR scenario: Q noise >> reward gap between modes.

Advantage weighting can't reliably separate modes when:
  - Good mode:    r ≈ 0
  - Mediocre:     r ≈ -0.5
  - Bad mode:     r ≈ -2.0
  - Q noise std = 3.0  (SNR ≈ 0.67)

Partial OT uses geometric proximity (concentrated expert cluster)
instead of noisy reward signal.
"""

import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.bandit_env import (
    generate_dataset, ground_truth_q, noisy_q,
    optimal_action, behavior_policy_sample,
)
from src.methods.bc import BehavioralCloning
from src.methods.kl_forward import ForwardKLPolicy
from src.methods.kl_reverse import ReverseKLPolicy
from src.methods.ot_partial import PartialOTPolicy
from src.methods.ot_unbalanced import UnbalancedOTPolicy
from src.methods.ot_wasserstein import WassersteinPolicy
from src.models.behavior_model import FittedBehaviorModel
from src.models.policy_net import GaussianPolicy, GMMPolicy
from src.training.trainer import Trainer
from src.visualization.plots import (
    make_state_grid, plot_reward_heatmaps, plot_training_curves,
)


def load_config(path: str = "configs/low_snr.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_policy(config: dict):
    policy_cfg = config.get("policy", {})
    if policy_cfg.get("type") == "gmm":
        return GMMPolicy(n_components=policy_cfg.get("n_components", 5))
    return GaussianPolicy()


def plot_snr_comparison(results, representative_states, env_config,
                        n_samples=500, save_path=None):
    """Comparison plot with behavior colored by distance from optimal."""
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
            ax.scatter(a_behav[:, 0], a_behav[:, 1], c=dist_from_opt,
                      cmap="RdYlGn_r", alpha=0.15, s=5, vmin=0, vmax=3)
            ax.scatter(a_policy[:, 0], a_policy[:, 1], c="C0", alpha=0.3, s=8, label="Policy")
            ax.scatter([a_opt[0]], [a_opt[1]], c="red", marker="*", s=300, zorder=10, label="Optimal")

            mu = policy.mean_action(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).detach().numpy()[0]
            ax.scatter([mu[0]], [mu[1]], c="blue", marker="x", s=150, zorder=10)

            if i == 0:
                ax.set_title(f"s=({s[0]:.0f},{s[1]:.0f})", fontsize=13)
            if j == 0:
                ax.set_ylabel(name, fontsize=12, fontweight="bold")
            ax.set_xlim(a_opt[0] - 4, a_opt[0] + 5)
            ax.set_ylim(a_opt[1] - 4, a_opt[1] + 5)
            ax.set_aspect("equal")
            if i == 0 and j == n_states - 1:
                ax.legend(fontsize=7, loc="upper right")

    plt.suptitle("Low SNR: Behavior (colored by dist) + Policy (blue)\n"
                 "Modes are close together → hard to distinguish by noisy Q",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def main():
    config = load_config()
    env_config = config["env"]
    train_config = config["training"]
    viz_config = config["viz"]
    corruption = config.get("corruption", {})

    seed = train_config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs("results", exist_ok=True)

    # Print SNR analysis
    modes = env_config.get("modes", [])
    print("=" * 60)
    print("LOW SNR SCENARIO")
    print("=" * 60)
    q_noise = corruption.get("q_noise_std", 0)
    for k, m in enumerate(modes):
        offset = np.array(m["offset"])
        approx_r = -np.sum(offset ** 2)
        snr = abs(approx_r) / q_noise if q_noise > 0 else float("inf")
        dist = np.linalg.norm(offset)
        q = "GOOD" if dist < 0.3 else ("MED" if dist < 0.8 else "BAD")
        print(f"  Mode {k+1}: offset={m['offset']}, r≈{approx_r:.1f}, SNR≈{snr:.2f} [{q}]")
    print(f"  Q noise std: {q_noise}")
    print(f"  Outlier fraction: {env_config.get('outlier_fraction', 0)}")
    print("=" * 60)

    # Generate dataset
    print("\nGenerating dataset...")
    dataset = generate_dataset(env_config, seed=seed)
    print(f"  N={dataset['states'].shape[0]}, Mean reward: {dataset['rewards'].mean():.4f}")

    # Fitted behavior model
    print("  Fitting behavior model...")
    behavior_model = FittedBehaviorModel(dataset, n_grid=5, n_components=5)

    # Noisy Q
    q_fn = partial(noisy_q, noise_std=q_noise)
    print(f"  Noisy Q (std={q_noise})")

    # Methods: all use same noisy Q
    methods = {}
    methods["BC"] = BehavioralCloning(make_policy(config), config.get("bc", {}))
    methods["Forward KL"] = ForwardKLPolicy(make_policy(config), config.get("kl_forward", {}), q_fn=q_fn)
    methods["Reverse KL"] = ReverseKLPolicy(make_policy(config), config.get("kl_reverse", {}), behavior_model, q_fn=q_fn)
    methods["Wasserstein"] = WassersteinPolicy(make_policy(config), config.get("ot_wasserstein", {}), env_config, q_fn=q_fn)
    methods["Partial OT"] = PartialOTPolicy(make_policy(config), config.get("ot_partial", {}), env_config, q_fn=q_fn)
    methods["Unbalanced OT"] = UnbalancedOTPolicy(make_policy(config), config.get("ot_unbalanced", {}), env_config, q_fn=q_fn)

    # Train
    results = {}
    trainers = {}
    for name, method in methods.items():
        print(f"\nTraining {name}...")
        trainer = Trainer(method, dataset, train_config)
        losses = trainer.train()
        results[name] = {"policy": method.policy, "losses": losses}
        trainers[name] = trainer
        print(f"  Final loss: {losses[-1]:.4f}")

    # Evaluate (ground-truth Q)
    print("\n" + "=" * 60)
    print("Mean Reward E[r(s, pi(s))] (ground-truth Q)")
    print("=" * 60)
    grid, _, _ = make_state_grid(n=50)
    s_grid = torch.tensor(grid, dtype=torch.float32)
    reward_table = {}
    for name, trainer in trainers.items():
        mean_r = trainer.evaluate_reward(s_grid, n_samples=100).mean().item()
        reward_table[name] = mean_r
        print(f"  {name}: {mean_r:.4f}")

    print("\n--- Ranking ---")
    for rank, (name, r) in enumerate(sorted(reward_table.items(), key=lambda x: -x[1]), 1):
        print(f"  {rank}. {name}: {r:.4f}")

    # Plots
    print("\nGenerating plots...")
    rep_states = viz_config.get("representative_states", [[0, 0], [1, 1], [-1, 2], [2, -1]])
    plot_snr_comparison(results, rep_states, env_config, save_path="results/low_snr_comparison.png")
    plot_reward_heatmaps(results, trainers, n_grid=50, save_path="results/low_snr_heatmaps.png")
    plot_training_curves(results, save_path="results/low_snr_training_curves.png")

    print("\nDone! Low SNR plots saved to results/")


if __name__ == "__main__":
    main()
