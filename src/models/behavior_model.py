import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from src.env.bandit_env import behavior_policy_log_prob_torch


class BehaviorModel:
    """Oracle behavior model using the known generative mixture."""

    def __init__(self, env_config: dict):
        self.config = env_config

    def log_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return behavior_policy_log_prob_torch(s, a, self.config)


class FittedBehaviorModel:
    """Fitted conditional GMM behavior model.

    Discretizes the state space into a grid and fits a separate GMM
    per cell. This introduces density estimation error, especially
    when the data contains outliers.
    """

    def __init__(self, dataset: dict, n_grid: int = 5, n_components: int = 3):
        self.n_grid = n_grid
        self.n_components = n_components
        self.gmms = {}
        self.fallback_gmm = None
        self._fit(dataset)

    def _state_to_cell(self, s: np.ndarray) -> tuple:
        """Map states to grid cell indices."""
        # Clip to [-3, 3] and discretize
        s_clipped = np.clip(s, -3, 3)
        idx = ((s_clipped + 3) / 6 * self.n_grid).astype(int)
        idx = np.clip(idx, 0, self.n_grid - 1)
        if s.ndim == 1:
            return (idx[0], idx[1])
        return list(zip(idx[:, 0], idx[:, 1]))

    def _fit(self, dataset: dict):
        states = dataset["states"].numpy()
        actions = dataset["actions"].numpy()

        # Fit a global fallback GMM
        self.fallback_gmm = GaussianMixture(
            n_components=self.n_components, covariance_type="full", random_state=0
        )
        self.fallback_gmm.fit(actions)

        # Fit per-cell GMMs
        cells = self._state_to_cell(states)
        cell_data = {}
        for i, cell in enumerate(cells):
            if cell not in cell_data:
                cell_data[cell] = []
            cell_data[cell].append(actions[i])

        for cell, acts in cell_data.items():
            acts = np.array(acts)
            if len(acts) >= 2 * self.n_components:
                gmm = GaussianMixture(
                    n_components=self.n_components, covariance_type="full", random_state=0
                )
                gmm.fit(acts)
                self.gmms[cell] = gmm

    def log_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        s_np = s.detach().cpu().numpy()
        a_np = a.detach().cpu().numpy()

        if s_np.ndim == 1:
            cell = self._state_to_cell(s_np)
            gmm = self.gmms.get(cell, self.fallback_gmm)
            lp = gmm.score_samples(a_np.reshape(1, -1))
            return torch.tensor(lp, dtype=torch.float32, device=s.device)

        cells = self._state_to_cell(s_np)
        log_probs = np.zeros(len(s_np))
        for i, cell in enumerate(cells):
            gmm = self.gmms.get(cell, self.fallback_gmm)
            log_probs[i] = gmm.score_samples(a_np[i : i + 1])[0]

        return torch.tensor(log_probs, dtype=torch.float32, device=s.device)
