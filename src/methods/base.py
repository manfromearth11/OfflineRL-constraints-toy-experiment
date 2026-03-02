from abc import ABC, abstractmethod

import torch

from src.models.policy_net import GaussianPolicy


class OfflineRLMethod(ABC):
    """Common interface for all offline RL methods."""

    def __init__(self, policy: GaussianPolicy, config: dict):
        self.policy = policy
        self.config = config

    @abstractmethod
    def compute_loss(self, batch: dict) -> torch.Tensor:
        """Return scalar loss for one gradient step."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for plots."""

    def train_step(self, batch: dict, optimizer: torch.optim.Optimizer) -> float:
        optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        optimizer.step()
        return loss.item()
