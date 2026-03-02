import torch
import torch.nn.functional as F

from src.methods.base import OfflineRLMethod
from src.models.policy_net import GaussianPolicy


class BehavioralCloning(OfflineRLMethod):

    def __init__(self, policy: GaussianPolicy, config: dict):
        super().__init__(policy, config)

    @property
    def name(self) -> str:
        return "BC"

    def compute_loss(self, batch: dict) -> torch.Tensor:
        s, a = batch["states"], batch["actions"]
        mu = self.policy.mean_action(s)
        return F.mse_loss(mu, a)
