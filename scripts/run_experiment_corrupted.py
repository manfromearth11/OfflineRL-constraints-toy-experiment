"""Corrupted scenario: outlier data + noisy Q + fitted behavior model.

This scenario is designed to show where OT methods have advantages over KL:
1. 25% outlier actions → Partial OT filters them naturally
2. Noisy Q → advantage weighting for KL becomes unreliable
3. Fitted behavior model → KL penalty is inaccurate
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

from src.env.bandit_env import generate_dataset, ground_truth_q, noisy_q
from src.methods.bc import BehavioralCloning
from src.methods.kl_forward import ForwardKLPolicy
from src.methods.kl_reverse import ReverseKLPolicy
from src.methods.ot_partial import PartialOTPolicy
from src.methods.ot_unbalanced import UnbalancedOTPolicy
from src.methods.ot_wasserstein import WassersteinPolicy
from src.models.behavior_model import BehaviorModel, FittedBehaviorModel
from src.models.policy_net import GaussianPolicy
from src.training.trainer import Trainer
from src.visualization.plots import (
    make_state_grid,
    plot_comparison_grid,
    plot_reward_heatmaps,
    plot_training_curves,
)


def load_config(path: str = "configs/corrupted.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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

    # Generate corrupted dataset
    print("=" * 60)
    print("CORRUPTED SCENARIO")
    print(f"  Outlier fraction: {env_config.get('outlier_fraction', 0)}")
    print(f"  Q noise std: {corruption.get('q_noise_std', 0)}")
    print(f"  Fitted behavior model: {corruption.get('use_fitted_behavior', False)}")
    print("=" * 60)

    print("\nGenerating corrupted dataset...")
    dataset = generate_dataset(env_config, seed=seed)
    print(f"  States: {dataset['states'].shape}")
    print(f"  Mean reward: {dataset['rewards'].mean():.4f}")

    # Behavior model: fitted (imperfect) or oracle
    if corruption.get("use_fitted_behavior", False):
        print("  Fitting behavior model (GMM)...")
        behavior_model = FittedBehaviorModel(dataset, n_grid=5, n_components=3)
        print("  Using FITTED behavior model (imperfect density)")
    else:
        behavior_model = BehaviorModel(env_config)
        print("  Using ORACLE behavior model")

    # Q-function: noisy or ground-truth
    q_noise_std = corruption.get("q_noise_std", 0.0)
    if q_noise_std > 0:
        q_fn = partial(noisy_q, noise_std=q_noise_std)
        print(f"  Using NOISY Q (std={q_noise_std})")
    else:
        q_fn = ground_truth_q
        print("  Using ground-truth Q")

    # Also generate a clean dataset for comparison evaluation
    clean_env_config = {k: v for k, v in env_config.items() if k != "outlier_fraction"}
    clean_dataset = generate_dataset(clean_env_config, seed=seed + 1)

    # Define methods
    methods = {}

    # BC
    methods["BC"] = BehavioralCloning(GaussianPolicy(), config.get("bc", {}))

    # KL with noisy Q + imperfect behavior model
    methods["Forward KL"] = ForwardKLPolicy(
        GaussianPolicy(), config.get("kl_forward", {}), q_fn=q_fn
    )
    methods["Reverse KL"] = ReverseKLPolicy(
        GaussianPolicy(), config.get("kl_reverse", {}), behavior_model, q_fn=q_fn
    )

    # OT methods: same noisy Q for fairness, but OT regularization is Q-independent
    methods["Wasserstein"] = WassersteinPolicy(
        GaussianPolicy(), config.get("ot_wasserstein", {}), env_config, q_fn=q_fn
    )
    methods["Partial OT"] = PartialOTPolicy(
        GaussianPolicy(), config.get("ot_partial", {}), env_config, q_fn=q_fn
    )
    methods["Unbalanced OT"] = UnbalancedOTPolicy(
        GaussianPolicy(), config.get("ot_unbalanced", {}), env_config, q_fn=q_fn
    )

    # Train all methods
    results = {}
    trainers = {}
    for name, method in methods.items():
        print(f"\nTraining {name}...")
        trainer = Trainer(method, dataset, train_config)
        losses = trainer.train()
        results[name] = {"policy": method.policy, "losses": losses}
        trainers[name] = trainer
        print(f"  Final loss: {losses[-1]:.4f}")

    # Evaluate mean reward (using GROUND-TRUTH Q, not noisy)
    print("\n" + "=" * 60)
    print("Mean Reward E[r(s, pi(s))] (ground-truth Q)")
    print("=" * 60)
    grid, _, _ = make_state_grid(n=viz_config.get("n_grid", 50))
    s_grid = torch.tensor(grid, dtype=torch.float32)
    reward_table = {}
    for name, trainer in trainers.items():
        mean_r = trainer.evaluate_reward(s_grid, n_samples=100).mean().item()
        reward_table[name] = mean_r
        print(f"  {name}: {mean_r:.4f}")

    # Sort by reward
    print("\n--- Ranking (best to worst) ---")
    for rank, (name, r) in enumerate(sorted(reward_table.items(), key=lambda x: -x[1]), 1):
        print(f"  {rank}. {name}: {r:.4f}")

    # Generate plots
    print("\nGenerating plots...")
    rep_states = viz_config.get("representative_states", [[0, 0], [1, 1], [-1, 2], [2, -1]])

    plot_comparison_grid(
        results, rep_states, env_config,
        n_samples=viz_config.get("n_plot_samples", 500),
        save_path="results/corrupted_comparison_grid.png",
    )

    plot_reward_heatmaps(
        results, trainers,
        n_grid=viz_config.get("n_grid", 50),
        save_path="results/corrupted_reward_heatmaps.png",
    )

    plot_training_curves(results, save_path="results/corrupted_training_curves.png")

    print("\nDone! Corrupted scenario plots saved to results/")


if __name__ == "__main__":
    main()
