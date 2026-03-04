import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.env.bandit_env import optimal_action, behavior_policy_sample


def make_state_grid(lo=-3, hi=3, n=50):
    """Create a 2D meshgrid of states."""
    x = np.linspace(lo, hi, n)
    y = np.linspace(lo, hi, n)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    return grid, xx, yy


def plot_comparison_grid(results: dict, representative_states: list,
                         env_config: dict, n_samples: int = 500, save_path: str = None):
    """
    Comparison grid: rows=methods, cols=representative states.
    Each cell shows behavior samples (gray), policy samples (color), optimal action (red star).
    """
    states = np.array(representative_states)
    method_names = list(results.keys())
    n_methods = len(method_names)
    n_states = len(states)

    fig, axes = plt.subplots(n_methods, n_states, figsize=(4 * n_states, 4 * n_methods))
    if n_methods == 1:
        axes = axes[np.newaxis, :]
    if n_states == 1:
        axes = axes[:, np.newaxis]

    for j, s in enumerate(states):
        # Behavior samples at this state
        s_tile = np.tile(s, (n_samples, 1))
        a_behav = behavior_policy_sample(s_tile, env_config)
        a_opt = optimal_action(s)

        for i, name in enumerate(method_names):
            ax = axes[i, j]
            policy = results[name]["policy"]

            # Policy samples
            with torch.no_grad():
                s_t = torch.tensor(s_tile, dtype=torch.float32)
                a_policy = policy.sample(s_t).numpy()

            ax.scatter(a_behav[:, 0], a_behav[:, 1], c="gray", alpha=0.15, s=5, label="Behavior")
            ax.scatter(a_policy[:, 0], a_policy[:, 1], c="C0", alpha=0.3, s=5, label="Policy")
            ax.scatter([a_opt[0]], [a_opt[1]], c="red", marker="*", s=200, zorder=5, label="Optimal")

            mu = policy.mean_action(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).detach().numpy()[0]
            ax.scatter([mu[0]], [mu[1]], c="C0", marker="x", s=100, zorder=5, label="Policy mean")

            if j == 0:
                ax.set_ylabel(name, fontsize=12, fontweight="bold")
            ax.set_xlim(a_opt[0] - 4, a_opt[0] + 6)
            ax.set_ylim(a_opt[1] - 4, a_opt[1] + 6)
            ax.set_aspect("equal")
            if i == 0 and j == n_states - 1:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_reward_heatmaps(results: dict, trainers: dict, n_grid: int = 50, save_path: str = None):
    """Reward heatmap E[r(s, pi(s))] for each method."""
    grid, xx, yy = make_state_grid(n=n_grid)
    s_grid = torch.tensor(grid, dtype=torch.float32)

    method_names = list(results.keys())
    n_methods = len(method_names)
    ncols = min(3, n_methods)
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    vmin, vmax = None, None
    reward_maps = {}
    for name in method_names:
        r_map = trainers[name].evaluate_reward(s_grid, n_samples=100).numpy().reshape(n_grid, n_grid)
        reward_maps[name] = r_map
        if vmin is None:
            vmin, vmax = r_map.min(), r_map.max()
        else:
            vmin = min(vmin, r_map.min())
            vmax = max(vmax, r_map.max())

    for idx, name in enumerate(method_names):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        im = ax.pcolormesh(xx, yy, reward_maps[name], cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax)

    # Hide unused axes
    for idx in range(n_methods, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_quiver_comparison(results: dict, n_grid: int = 20, save_path: str = None):
    """Quiver plot: policy mean actions vs optimal actions."""
    grid, xx, yy = make_state_grid(n=n_grid)
    s_grid = torch.tensor(grid, dtype=torch.float32)
    a_opt = optimal_action(grid)

    method_names = list(results.keys())
    n_methods = len(method_names)
    ncols = min(3, n_methods)
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, name in enumerate(method_names):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        policy = results[name]["policy"]

        with torch.no_grad():
            mu = policy.mean_action(s_grid).numpy()

        # Error arrows (difference from optimal)
        err = np.linalg.norm(mu - a_opt, axis=-1)
        ax.quiver(grid[:, 0], grid[:, 1], a_opt[:, 0], a_opt[:, 1],
                  color="gray", alpha=0.3, scale=30, label="Optimal")
        q = ax.quiver(grid[:, 0], grid[:, 1], mu[:, 0], mu[:, 1],
                      err, cmap="coolwarm", scale=30, label="Policy")
        plt.colorbar(q, ax=ax, label="||error||")
        ax.set_aspect("equal")

    for idx in range(n_methods, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_training_curves(results: dict, save_path: str = None):
    """Training loss curves for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in results.items():
        losses = data["losses"]
        ax.plot(losses, label=name, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_partial_ot_sweep(policy_class, policy_kwargs, env_config, dataset, train_config,
                          mass_fractions=(0.3, 0.5, 0.7, 1.0),
                          representative_state=(0, 0), n_samples=500, save_path=None):
    """Show how partial OT policy changes as mass_fraction varies."""
    from src.models.policy_net import GaussianPolicy
    from src.methods.ppl import PartialOTPolicy
    from src.training.trainer import Trainer

    s = np.array(representative_state)
    s_tile = np.tile(s, (n_samples, 1))
    a_behav = behavior_policy_sample(s_tile, env_config)
    a_opt = optimal_action(s)

    fig, axes = plt.subplots(1, len(mass_fractions), figsize=(5 * len(mass_fractions), 5))
    if len(mass_fractions) == 1:
        axes = [axes]

    for idx, m in enumerate(mass_fractions):
        ax = axes[idx]
        policy = GaussianPolicy()
        ot_config = {**train_config.get("ot_partial", {}), "mass_fraction": m}
        method = PartialOTPolicy(policy, ot_config, env_config)
        trainer = Trainer(method, dataset, train_config)
        print(f"  Training Partial OT (m={m})...")
        trainer.train(n_epochs=train_config.get("n_epochs", 200))

        with torch.no_grad():
            s_t = torch.tensor(s_tile, dtype=torch.float32)
            a_policy = policy.sample(s_t).numpy()

        ax.scatter(a_behav[:, 0], a_behav[:, 1], c="gray", alpha=0.15, s=5, label="Behavior")
        ax.scatter(a_policy[:, 0], a_policy[:, 1], c="C0", alpha=0.3, s=5, label="Policy")
        ax.scatter([a_opt[0]], [a_opt[1]], c="red", marker="*", s=200, zorder=5, label="Optimal")
        ax.set_xlim(a_opt[0] - 4, a_opt[0] + 6)
        ax.set_ylim(a_opt[1] - 4, a_opt[1] + 6)
        ax.set_aspect("equal")
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()
