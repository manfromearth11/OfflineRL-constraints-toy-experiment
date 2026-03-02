"""Reward-free Imitation Learning scenario.

Key insight: Without Q-function, KL advantage weighting = uniform weights → BC.
Partial OT uses geometric cost to select the concentrated expert cluster,
ignoring spread-out non-expert and random demonstrations.

This is where OT methods have a clear advantage over KL/BC methods.
"""

import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.bandit_env import generate_dataset, ground_truth_q, optimal_action, behavior_policy_sample
from src.methods.bc import BehavioralCloning
from src.methods.kl_forward import ForwardKLPolicy
from src.methods.kl_reverse import ReverseKLPolicy
from src.methods.ot_partial import PartialOTPolicy
from src.methods.ot_unbalanced import UnbalancedOTPolicy
from src.methods.ot_wasserstein import WassersteinPolicy
from src.models.behavior_model import FittedBehaviorModel
from src.models.policy_net import GaussianPolicy, GMMPolicy
from src.training.trainer import Trainer
from src.visualization.plots import make_state_grid, plot_reward_heatmaps, plot_training_curves


def load_config(path: str = "configs/reward_free.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_policy(config: dict):
    policy_cfg = config.get("policy", {})
    if policy_cfg.get("type") == "gmm":
        return GMMPolicy(n_components=policy_cfg.get("n_components", 5))
    return GaussianPolicy()


def zero_q(s, a):
    """Zero Q-function: no reward information available."""
    if isinstance(s, torch.Tensor):
        return torch.zeros(s.shape[0], device=s.device)
    return np.zeros(s.shape[0])


def plot_reward_free_comparison(results, representative_states, env_config,
                                 n_samples=500, save_path=None):
    """Comparison grid for reward-free scenario."""
    import matplotlib.pyplot as plt
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

        # Generate behavior samples for visualization
        rng = np.random.default_rng(j + 42)
        a_behav = behavior_policy_sample(s_tile, env_config, rng)

        for i, name in enumerate(method_names):
            ax = axes[i, j]
            policy = results[name]["policy"]

            with torch.no_grad():
                s_t = torch.tensor(s_tile, dtype=torch.float32)
                a_policy = policy.sample(s_t).numpy()

            # Behavior: colored by distance from optimal
            dist_from_opt = np.linalg.norm(a_behav - a_opt, axis=-1)
            ax.scatter(a_behav[:, 0], a_behav[:, 1], c=dist_from_opt,
                      cmap="RdYlGn_r", alpha=0.15, s=5, vmin=0, vmax=5)
            ax.scatter(a_policy[:, 0], a_policy[:, 1], c="C0", alpha=0.3, s=8, label="Policy")
            ax.scatter([a_opt[0]], [a_opt[1]], c="red", marker="*", s=300, zorder=10, label="Optimal")

            mu = policy.mean_action(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).detach().numpy()[0]
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

    plt.suptitle("Reward-Free IL: Behavior (colored by dist from optimal) + Policy (blue)",
                 fontsize=14, fontweight="bold")
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

    modes = env_config.get("modes", [])
    print("=" * 60)
    print("REWARD-FREE IMITATION LEARNING SCENARIO")
    print("  No Q-function available → pure distribution matching")
    print(f"  {len(modes)} demonstration modes:")
    for k, m in enumerate(modes):
        dist = np.linalg.norm(m["offset"])
        q = "EXPERT" if dist < 0.5 else ("SUB-EXPERT" if dist < 1.5 else "RANDOM")
        print(f"    Mode {k+1}: offset={m['offset']}, std={m['std']:.2f}, w={m['weight']:.2f} [{q}]")
    print("=" * 60)

    # Generate dataset
    print("\nGenerating dataset...")
    dataset = generate_dataset(env_config, seed=seed)
    print(f"  N={dataset['states'].shape[0]}, Mean reward: {dataset['rewards'].mean():.4f}")

    # Fitted behavior model (imperfect)
    print("  Fitting behavior model...")
    behavior_model = FittedBehaviorModel(dataset, n_grid=5, n_components=5)

    # Q-function: ZERO (no reward info for KL methods)
    # OT methods also get zero Q → their loss is pure OT regularization
    q_fn = zero_q
    print("  Q-function: ZERO (reward-free)")

    # Define methods
    methods = {}

    # BC: MSE to all data (no Q needed)
    methods["BC"] = BehavioralCloning(make_policy(config), config.get("bc", {}))

    # KL methods: with zero Q, advantage = 0 for all → uniform weights → BC
    methods["Forward KL"] = ForwardKLPolicy(
        make_policy(config), config.get("kl_forward", {}), q_fn=q_fn
    )
    methods["Reverse KL"] = ReverseKLPolicy(
        make_policy(config), config.get("kl_reverse", {}), behavior_model, q_fn=q_fn
    )

    # OT methods: pure distribution matching, different regularizations
    # With Q=0, loss = 0 + λ*OT_cost → minimizes OT distance to behavior
    # But the key: Partial OT only matches nearest 30%!
    methods["Wasserstein"] = WassersteinPolicy(
        make_policy(config), config.get("ot_wasserstein", {}), env_config, q_fn=q_fn
    )
    methods["Partial OT"] = PartialOTPolicy(
        make_policy(config), config.get("ot_partial", {}), env_config, q_fn=q_fn
    )
    methods["Unbalanced OT"] = UnbalancedOTPolicy(
        make_policy(config), config.get("ot_unbalanced", {}), env_config, q_fn=q_fn
    )

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

    # Evaluate using GROUND-TRUTH reward (not available during training!)
    print("\n" + "=" * 60)
    print("Mean Reward E[r(s, pi(s))] (ground-truth, NOT used during training)")
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

    plot_reward_free_comparison(
        results, rep_states, env_config,
        n_samples=viz_config.get("n_plot_samples", 500),
        save_path="results/reward_free_comparison.png",
    )
    plot_reward_heatmaps(results, trainers, n_grid=50, save_path="results/reward_free_heatmaps.png")
    plot_training_curves(results, save_path="results/reward_free_training_curves.png")

    print("\nDone! Reward-free scenario plots saved to results/")


if __name__ == "__main__":
    main()
