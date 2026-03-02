import torch

from src.env.bandit_env import ground_truth_q
from src.methods.base import OfflineRLMethod
from src.models.policy_net import GaussianPolicy


class ReverseKLPolicy(OfflineRLMethod):
    """Reverse KL / AWR (mode-seeking): weighted MLE + explicit reverse KL penalty."""

    def __init__(self, policy: GaussianPolicy, config: dict, behavior_model, q_fn=None):
        super().__init__(policy, config)
        self.alpha = config.get("alpha", 1.0)
        self.beta = config.get("beta", 0.1)
        self.behavior_model = behavior_model
        self.q_fn = q_fn or ground_truth_q

    @property
    def name(self) -> str:
        return "Reverse KL"

    def compute_loss(self, batch: dict) -> torch.Tensor:
        s, a = batch["states"], batch["actions"]

        q_values = self.q_fn(s, a)
        advantages = q_values - q_values.mean()

        log_weights = advantages / self.alpha
        log_weights = log_weights - log_weights.max()
        weights = torch.exp(log_weights)
        weights = weights / weights.sum()

        log_probs = self.policy.log_prob(s, a)
        mle_loss = -(weights.detach() * log_probs).sum()

        a_sampled = self.policy.sample(s)
        kl_penalty = (self.policy.log_prob(s, a_sampled) - self.behavior_model.log_prob(s, a_sampled)).mean()

        return mle_loss + self.beta * kl_penalty
