import torch

from src.env.bandit_env import ground_truth_q
from src.methods.base import OfflineRLMethod
from src.models.policy_net import GaussianPolicy


class L2ConstraintPolicy(OfflineRLMethod):
    """Q maximization with simple action L2 anchor to dataset actions."""

    def __init__(self, policy: GaussianPolicy, config: dict, q_fn=None):
        super().__init__(policy, config)
        self.lam = float(config.get("lam", 1.0))
        self.anchor_mode = config.get("anchor_mode", "dataset_action")
        if self.anchor_mode != "dataset_action":
            raise ValueError(
                f"Unsupported anchor_mode='{self.anchor_mode}'. "
                "Only 'dataset_action' is currently supported."
            )
        self.q_fn = q_fn or ground_truth_q

    @property
    def name(self) -> str:
        return "L2 Constraint"

    def compute_loss(self, batch: dict) -> torch.Tensor:
        s = batch["states"]
        a_beta = batch["actions"]
        a_pi = self.policy.sample(s)

        q_term = -self.q_fn(s, a_pi).mean()
        l2_term = (a_pi - a_beta).pow(2).sum(dim=-1).mean()
        return q_term + self.lam * l2_term
