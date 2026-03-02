import torch
from torch.utils.data import DataLoader, TensorDataset

from src.methods.base import OfflineRLMethod


class Trainer:
    """Generic training loop for all offline RL methods."""

    def __init__(self, method: OfflineRLMethod, dataset: dict, config: dict):
        self.method = method
        self.dataset = dataset
        self.config = config
        self.optimizer = torch.optim.Adam(
            method.policy.parameters(), lr=config.get("lr", 1e-3)
        )

        ds = TensorDataset(dataset["states"], dataset["actions"], dataset["rewards"])
        self.dataloader = DataLoader(
            ds, batch_size=config.get("batch_size", 256), shuffle=True, drop_last=True
        )

    def train(self, n_epochs: int = None) -> list:
        n_epochs = n_epochs or self.config.get("n_epochs", 200)
        epoch_losses = []

        for epoch in range(n_epochs):
            batch_losses = []
            for states, actions, rewards in self.dataloader:
                batch = {"states": states, "actions": actions, "rewards": rewards}
                loss = self.method.train_step(batch, self.optimizer)
                batch_losses.append(loss)
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)

            if (epoch + 1) % 50 == 0:
                print(f"  [{self.method.name}] Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}")

        return epoch_losses

    @torch.no_grad()
    def evaluate_reward(self, s_grid: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """Compute E_{a~pi}[r(s,a)] for each state in the grid."""
        from src.env.bandit_env import ground_truth_q

        B = s_grid.shape[0]
        s_expand = s_grid.unsqueeze(1).expand(B, n_samples, 2).reshape(B * n_samples, 2)
        a = self.method.policy.sample(s_expand)
        q = ground_truth_q(s_expand, a).reshape(B, n_samples)
        return q.mean(dim=1)
