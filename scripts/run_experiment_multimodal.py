"""Multi-modal scenario: many behavior modes + outliers + noisy Q + MDN policy.

This demonstrates OT advantages over KL:
1. 6 behavior modes (2 good + 2 mediocre + 2 bad) + 20% outliers
2. Noisy Q → advantage weighting can't reliably separate close modes
3. OT geometric proximity naturally selects nearest good modes
4. Partial OT with m=0.3 transports only to the best 30% of data
5. MDN policy preserves multi-modal structure
"""

import os
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.bandit_env import (
    generate_dataset,
    ground_truth_q,
    noisy_q,
    optimal_action,
    behavior_policy_sample,
)
from src.methods.bc import BehavioralCloning
from src.methods.kl_forward import ForwardKLPolicy
from src.methods.kl_reverse import ReverseKLPolicy
from src.methods.ot_partial import PartialOTPolicy
from src.methods.ot_unbalanced import UnbalancedOTPolicy
from src.methods.ot_wasserstein import WassersteinPolicy
from src.models.behavior_model import BehaviorModel, FittedBehaviorModel
from src.models.policy_net import GaussianPolicy, GMMPolicy
from src.training.trainer import Trainer
from src.visualization.plots import make_state_grid


def load_config(path: str = "configs/multimodal.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_policy(config: dict):
    """Create policy based on config."""
    policy_cfg = config.get("policy", {})
    if policy_cfg.get("type") == "gmm":
        return GMMPolicy(n_components=policy_cfg.get("n_components", 5))
    return GaussianPolicy()


def plot_multimodal_comparison(results, representative_states, env_config,
                               n_samples=500, save_path=None):
    """Comparison grid with mode structure visualization."""
    states = np.array(representative_states)
    method_names = list(results.keys())
    n_methods = len(method_names)
    n_states = len(states)

    fig, axes = plt.subplots(n_methods, n_states, figsize=(5 * n_states, 5 * n_methods))
    if n_methods == 1:
        axes = axes[np.newaxis, :]

    colors_mode = {
        "good": "green",
        "mediocre": "orange",
        "bad": "red",
    }

    for j, s in enumerate(states):
        s_tile = np.tile(s, (n_samples, 1))
        a_opt = optimal_action(s)

        # Draw behavior samples colored by their mode quality
        modes = env_config.get("modes", [])
        if modes:
            a_behav_by_mode = {}
            rng = np.random.default_rng(j)
            for k, mode in enumerate(modes):
                offset = np.array(mode["offset"])
                std = mode["std"]
                n_mode = int(n_samples * mode["weight"])
                a_mode = a_opt + offset + rng.normal(0, std, (n_mode, 2))
                quality = mode.get("quality", "mediocre")
                # Auto-classify by offset distance if quality not specified
                dist = np.linalg.norm(offset)
                if dist < 1.0:
                    quality = "good"
                elif dist < 2.5:
                    quality = "mediocre"
                else:
                    quality = "bad"
                if quality not in a_behav_by_mode:
                    a_behav_by_mode[quality] = []
                a_behav_by_mode[quality].append(a_mode)

        for i, name in enumerate(method_names):
            ax = axes[i, j]
            policy = results[name]["policy"]

            # Draw behavior modes
            if modes:
                for quality, mode_samples in a_behav_by_mode.items():
                    all_samples = np.concatenate(mode_samples, axis=0)
                    ax.scatter(all_samples[:, 0], all_samples[:, 1],
                              c=colors_mode.get(quality, "gray"),
                              alpha=0.1, s=3, label=f"Behav ({quality})")

            # Draw outlier region hint
            if env_config.get("outlier_fraction", 0) > 0:
                n_outlier = int(n_samples * env_config["outlier_fraction"])
                rng_o = np.random.default_rng(j + 100)
                a_outlier = a_opt + rng_o.uniform(-5, 5, (n_outlier, 2))
                ax.scatter(a_outlier[:, 0], a_outlier[:, 1],
                          c="gray", alpha=0.05, s=2, marker="x")

            # Policy samples
            with torch.no_grad():
                s_t = torch.tensor(s_tile, dtype=torch.float32)
                a_policy = policy.sample(s_t).numpy()

            ax.scatter(a_policy[:, 0], a_policy[:, 1], c="C0", alpha=0.25, s=8, label="Policy")
            ax.scatter([a_opt[0]], [a_opt[1]], c="red", marker="*", s=300, zorder=10, label="Optimal")

            mu = policy.mean_action(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).detach().numpy()[0]
            ax.scatter([mu[0]], [mu[1]], c="blue", marker="x", s=150, zorder=10, label="Policy mean")

            if i == 0:
                ax.set_title(f"s=({s[0]:.0f},{s[1]:.0f})", fontsize=13)
            if j == 0:
                ax.set_ylabel(name, fontsize=13, fontweight="bold")
            ax.set_xlim(a_opt[0] - 6, a_opt[0] + 6)
            ax.set_ylim(a_opt[1] - 6, a_opt[1] + 6)
            ax.set_aspect("equal")
            if i == 0 and j == n_states - 1:
                ax.legend(fontsize=6, loc="upper right")

    plt.suptitle("Multi-Modal Scenario: Behavior Modes + Policy", fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_mode_coverage(results, env_config, test_state=(0, 0), n_samples=1000, save_path=None):
    """Bar chart showing how much each method covers each behavior mode."""
    s = np.array(test_state)
    a_opt = optimal_action(s)
    modes = env_config.get("modes", [])
    if not modes:
        return

    method_names = list(results.keys())
    mode_labels = []
    mode_centers = []
    for k, mode in enumerate(modes):
        offset = np.array(mode["offset"])
        dist = np.linalg.norm(offset)
        if dist < 1.0:
            q = "Good"
        elif dist < 2.5:
            q = "Med"
        else:
            q = "Bad"
        mode_labels.append(f"M{k+1} ({q})")
        mode_centers.append(a_opt + offset)
    mode_centers = np.array(mode_centers)

    # For each method, count how many policy samples land near each mode center
    coverage = {}
    for name in method_names:
        policy = results[name]["policy"]
        s_tile = np.tile(s, (n_samples, 1))
        with torch.no_grad():
            a = policy.sample(torch.tensor(s_tile, dtype=torch.float32)).numpy()

        # Assign each sample to nearest mode
        dists = np.linalg.norm(a[:, None, :] - mode_centers[None, :, :], axis=-1)  # (N, K)
        nearest = dists.argmin(axis=1)

        counts = np.zeros(len(modes))
        for k in range(len(modes)):
            # Only count if within 2*std of mode center
            mask = nearest == k
            if mask.sum() > 0:
                in_range = dists[mask, k] < 2 * modes[k]["std"]
                counts[k] = in_range.sum() / n_samples

        coverage[name] = counts

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(modes))
    width = 0.12
    for i, name in enumerate(method_names):
        ax.bar(x + i * width, coverage[name], width, label=name, alpha=0.8)

    ax.set_xlabel("Behavior Mode")
    ax.set_ylabel("Coverage (fraction of samples)")
    ax.set_title(f"Mode Coverage at s=({test_state[0]},{test_state[1]})")
    ax.set_xticks(x + width * (len(method_names) - 1) / 2)
    ax.set_xticklabels(mode_labels)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

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

    # Print setup
    modes = env_config.get("modes", [])
    print("=" * 60)
    print("MULTI-MODAL SCENARIO")
    print(f"  {len(modes)} behavior modes + {env_config.get('outlier_fraction', 0)*100:.0f}% outliers")
    print(f"  Q noise std: {corruption.get('q_noise_std', 0)}")
    print(f"  Policy type: {config.get('policy', {}).get('type', 'gaussian')}")
    for k, m in enumerate(modes):
        dist = np.linalg.norm(m["offset"])
        q = "GOOD" if dist < 1.0 else ("MED" if dist < 2.5 else "BAD")
        print(f"    Mode {k+1}: offset={m['offset']}, std={m['std']}, w={m['weight']:.2f} [{q}]")
    print("=" * 60)

    # Generate dataset
    print("\nGenerating dataset...")
    dataset = generate_dataset(env_config, seed=seed)
    print(f"  N={dataset['states'].shape[0]}, Mean reward: {dataset['rewards'].mean():.4f}")

    # Behavior model
    if corruption.get("use_fitted_behavior", False):
        print("  Fitting behavior model (GMM)...")
        behavior_model = FittedBehaviorModel(dataset, n_grid=5, n_components=5)
    else:
        behavior_model = BehaviorModel(env_config)

    # Q-function
    q_noise_std = corruption.get("q_noise_std", 0.0)
    q_fn = partial(noisy_q, noise_std=q_noise_std) if q_noise_std > 0 else ground_truth_q
    print(f"  Q noise: {q_noise_std}")

    # Define methods (all use same noisy Q)
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

    # Generate plots
    print("\nGenerating plots...")
    rep_states = viz_config.get("representative_states", [[0, 0], [1, 1], [-1, 2], [2, -1]])

    plot_multimodal_comparison(
        results, rep_states, env_config,
        n_samples=viz_config.get("n_plot_samples", 500),
        save_path="results/multimodal_comparison_grid.png",
    )

    plot_mode_coverage(
        results, env_config, test_state=rep_states[0],
        save_path="results/multimodal_mode_coverage.png",
    )

    # Reward heatmaps
    from src.visualization.plots import plot_reward_heatmaps, plot_training_curves
    plot_reward_heatmaps(results, trainers, n_grid=50, save_path="results/multimodal_reward_heatmaps.png")
    plot_training_curves(results, save_path="results/multimodal_training_curves.png")

    print("\nDone! Multi-modal scenario plots saved to results/")


if __name__ == "__main__":
    main()
