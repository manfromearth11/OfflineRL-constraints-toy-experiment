import copy
import warnings

import torch
import torch.nn as nn

from src.env.bandit_env import ground_truth_q
from src.methods.base import OfflineRLMethod
from src.models.policy_net import GaussianPolicy


class PotentialMLP(nn.Module):
    """Potential network f_omega(s, a)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        output_nonnegative: bool = True,
    ):
        super().__init__()
        self.output_nonnegative = output_nonnegative
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([s, a], dim=-1)).squeeze(-1)
        if self.output_nonnegative:
            out = self.softplus(out)
        return out


class PPLPolicy(OfflineRLMethod):
    """Potential-based policy optimization.

    Policy step:
        L_pi = -E[Q(s,a_pi)] - lambda * E[f_omega(s,a_pi)]

    Potential step (gradient descent form):
        L_f = E[f_omega(s,a_pi)] - w * E[f_omega(s,a_beta)] + wd * ||omega||^2

    Note: this matches maximizing (w * E_beta[f] - E_pi[f]) for the potential player.
    """

    def __init__(self, policy: GaussianPolicy, config: dict, env_config: dict, q_fn=None):
        super().__init__(policy, config)
        self.lam = float(config.get("lam", 1.0))

        # Backward compatibility with older key. Prefer w for clear semantics.
        self.w = float(config.get("w", config.get("mass_fraction", 1.0)))
        if self.w < 1.0:
            warnings.warn(
                f"PPLPolicy received w={self.w:.3f} (<1). "
                "For standard PPL-style interpretation, w>=1 is recommended.",
                RuntimeWarning,
            )

        self.pot_hidden_dim = int(config.get("pot_hidden_dim", 128))
        self.pot_lr = float(config.get("pot_lr", 1e-4))
        self.pot_steps_per_batch = int(config.get("pot_steps_per_batch", 1))
        self.pot_weight_decay = float(config.get("pot_weight_decay", 0.0))
        self.pot_output_nonnegative = bool(config.get("pot_output_nonnegative", True))

        self.q_fn = q_fn or ground_truth_q
        self._potential = None
        self._potential_optimizer = None

    @property
    def name(self) -> str:
        return f"PPL (w={self.w:g})"

    def _ensure_potential(self, s: torch.Tensor, a: torch.Tensor) -> None:
        if self._potential is not None:
            return
        self._potential = PotentialMLP(
            state_dim=s.shape[-1],
            action_dim=a.shape[-1],
            hidden_dim=self.pot_hidden_dim,
            output_nonnegative=self.pot_output_nonnegative,
        ).to(s.device)
        self._potential_optimizer = torch.optim.Adam(self._potential.parameters(), lr=self.pot_lr)

    def _potential_l2(self) -> torch.Tensor:
        reg = torch.tensor(0.0, device=next(self._potential.parameters()).device)
        for p in self._potential.parameters():
            reg = reg + p.pow(2).sum()
        return reg

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """Policy-side objective with frozen potential."""
        s = batch["states"]
        a_pi = self.policy.sample(s)
        q_vals = self.q_fn(s, a_pi).mean()

        self._ensure_potential(s, batch["actions"])
        for p in self._potential.parameters():
            p.requires_grad_(False)
        pot_vals = self._potential(s, a_pi).mean()
        for p in self._potential.parameters():
            p.requires_grad_(True)

        return -q_vals - self.lam * pot_vals

    def train_step(self, batch: dict, optimizer: torch.optim.Optimizer) -> float:
        s, a_beta = batch["states"], batch["actions"]
        self._ensure_potential(s, a_beta)

        # 1) Potential update(s): minimize E_pi[f] - w E_beta[f] + wd ||omega||^2.
        for _ in range(self.pot_steps_per_batch):
            with torch.no_grad():
                a_pi_det = self.policy.sample(s).detach()

            pot_pi = self._potential(s, a_pi_det).mean()
            pot_beta = self._potential(s, a_beta).mean()
            reg = self.pot_weight_decay * self._potential_l2()
            pot_loss = pot_pi - self.w * pot_beta + reg

            self._potential_optimizer.zero_grad()
            pot_loss.backward()
            self._potential_optimizer.step()

        # 2) Policy update with frozen potential.
        for p in self._potential.parameters():
            p.requires_grad_(False)

        optimizer.zero_grad()
        a_pi = self.policy.sample(s)
        q_vals = self.q_fn(s, a_pi).mean()
        pot_vals = self._potential(s, a_pi).mean()
        policy_loss = -q_vals - self.lam * pot_vals
        policy_loss.backward()
        optimizer.step()

        for p in self._potential.parameters():
            p.requires_grad_(True)

        return float(policy_loss.item())


def make_ppl_config(base: dict) -> dict:
    """Small helper for callers that want explicit migration semantics."""
    cfg = copy.deepcopy(base)
    if "w" not in cfg and "mass_fraction" in cfg:
        cfg["w"] = cfg["mass_fraction"]
    return cfg


class PartialOTPolicy(PPLPolicy):
    """Compatibility alias for old method naming."""

    @property
    def name(self) -> str:
        return f"Partial OT (PPL-style, w={self.w:g})"
