import torch

from src.env.bandit_env import ground_truth_q
from src.methods.base import OfflineRLMethod
from src.models.policy_net import GaussianPolicy


class ForwardKLPolicy(OfflineRLMethod):
    """Forward KL (mass-covering): advantage-weighted MLE."""

    def __init__(self, policy: GaussianPolicy, config: dict, q_fn=None):
        super().__init__(policy, config)
        self.alpha = config.get("alpha", 1.0)
        self.q_fn = q_fn or ground_truth_q

    @property
    def name(self) -> str:
        return "Forward KL"

    def compute_loss(self, batch: dict) -> torch.Tensor:
        s, a = batch["states"], batch["actions"]

        q_values = self.q_fn(s, a)
        advantages = q_values - q_values.mean()

        log_weights = advantages / self.alpha
        log_weights = log_weights - log_weights.max()
        weights = torch.exp(log_weights)
        weights = weights / weights.sum()

        log_probs = self.policy.log_prob(s, a)
        return -(weights.detach() * log_probs).sum()
