"""Main experiment script: train all methods and generate comparison plots."""

import os
import sys
from pathlib import Path

import yaml
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.bandit_env import generate_dataset
from src.models.policy_net import GaussianPolicy
from src.models.behavior_model import BehaviorModel
from src.methods.bc import BehavioralCloning
from src.methods.kl_forward import ForwardKLPolicy
from src.methods.kl_reverse import ReverseKLPolicy
from src.methods.ot_wasserstein import WassersteinPolicy
from src.methods.ot_partial import PartialOTPolicy
from src.methods.ot_unbalanced import UnbalancedOTPolicy
from src.training.trainer import Trainer
from src.visualization.plots import (
    plot_comparison_grid,
    plot_reward_heatmaps,
    plot_quiver_comparison,
    plot_training_curves,
    plot_partial_ot_sweep,
)


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    env_config = config["env"]
    train_config = config["training"]
    viz_config = config["viz"]

    seed = train_config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs("results", exist_ok=True)

    # Generate dataset
    print("Generating dataset...")
    dataset = generate_dataset(env_config, seed=seed)
    print(f"  States: {dataset['states'].shape}, Actions: {dataset['actions'].shape}")
    print(f"  Mean reward: {dataset['rewards'].mean():.4f}")

    behavior_model = BehaviorModel(env_config)

    # Define methods
    def make_methods():
        methods = {}

        # BC
        policy_bc = GaussianPolicy()
        methods["BC"] = BehavioralCloning(policy_bc, config.get("bc", {}))

        # Forward KL
        policy_fkl = GaussianPolicy()
        methods["Forward KL"] = ForwardKLPolicy(policy_fkl, config.get("kl_forward", {}))

        # Reverse KL
        policy_rkl = GaussianPolicy()
        methods["Reverse KL"] = ReverseKLPolicy(policy_rkl, config.get("kl_reverse", {}), behavior_model)

        # Wasserstein
        policy_w2 = GaussianPolicy()
        methods["Wasserstein"] = WassersteinPolicy(policy_w2, config.get("ot_wasserstein", {}), env_config)

        # Partial OT
        policy_pot = GaussianPolicy()
        methods["Partial OT"] = PartialOTPolicy(policy_pot, config.get("ot_partial", {}), env_config)

        # Unbalanced OT
        policy_uot = GaussianPolicy()
        methods["Unbalanced OT"] = UnbalancedOTPolicy(policy_uot, config.get("ot_unbalanced", {}), env_config)

        return methods

    methods = make_methods()

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

    # Evaluate mean reward
    print("\n--- Mean Reward E[r(s, pi(s))] ---")
    from src.visualization.plots import make_state_grid
    grid, _, _ = make_state_grid(n=viz_config.get("n_grid", 50))
    s_grid = torch.tensor(grid, dtype=torch.float32)
    for name, trainer in trainers.items():
        mean_r = trainer.evaluate_reward(s_grid, n_samples=100).mean().item()
        print(f"  {name}: {mean_r:.4f}")

    # Generate plots
    print("\nGenerating plots...")

    rep_states = viz_config.get("representative_states", [[0, 0], [1, 1], [-1, 2], [2, -1]])
    plot_comparison_grid(results, rep_states, env_config,
                         n_samples=viz_config.get("n_plot_samples", 500),
                         save_path="results/comparison_grid.png")

    plot_reward_heatmaps(results, trainers,
                         n_grid=viz_config.get("n_grid", 50),
                         save_path="results/reward_heatmaps.png")

    plot_quiver_comparison(results, n_grid=20, save_path="results/quiver_comparison.png")

    plot_training_curves(results, save_path="results/training_curves.png")

    print("\nRunning Partial OT sweep...")
    plot_partial_ot_sweep(
        GaussianPolicy, {}, env_config, dataset, {**train_config, **config.get("ot_partial", {})},
        mass_fractions=(0.3, 0.5, 0.7, 1.0),
        representative_state=rep_states[0],
        save_path="results/partial_ot_sweep.png",
    )

    print("\nDone! All plots saved to results/")


if __name__ == "__main__":
    main()
