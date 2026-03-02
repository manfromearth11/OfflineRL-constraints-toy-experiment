import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy: pi_theta(a|s) = N(mu(s), diag(sigma(s)^2))."""

    def __init__(self, state_dim: int = 2, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, s: torch.Tensor):
        h = self.trunk(s)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-4.0, 2.0)
        return mu, log_sigma

    def dist(self, s: torch.Tensor) -> Independent:
        mu, log_sigma = self.forward(s)
        return Independent(Normal(mu, log_sigma.exp()), 1)

    def sample(self, s: torch.Tensor, n: int = 1) -> torch.Tensor:
        d = self.dist(s)
        if n == 1:
            return d.rsample()
        return d.rsample((n,))

    def log_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.dist(s).log_prob(a)

    def mean_action(self, s: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(s)
        return mu


class GMMPolicy(nn.Module):
    """Gaussian Mixture Model policy (MDN): pi_theta(a|s) = sum_k w_k(s) * N(mu_k(s), sigma_k(s)^2).

    Can represent multi-modal action distributions.
    """

    def __init__(self, state_dim: int = 2, action_dim: int = 2,
                 hidden_dim: int = 64, n_components: int = 5):
        super().__init__()
        self.action_dim = action_dim
        self.n_components = n_components

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Per-component outputs
        self.logits_head = nn.Linear(hidden_dim, n_components)  # mixing logits
        self.mu_head = nn.Linear(hidden_dim, n_components * action_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, n_components * action_dim)

    def forward(self, s: torch.Tensor):
        """Returns (logits, mus, log_sigmas) for the mixture."""
        h = self.trunk(s)
        logits = self.logits_head(h)  # (B, K)
        mus = self.mu_head(h).view(*s.shape[:-1], self.n_components, self.action_dim)  # (B, K, D)
        log_sigmas = self.log_sigma_head(h).view(
            *s.shape[:-1], self.n_components, self.action_dim
        ).clamp(-4.0, 2.0)  # (B, K, D)
        return logits, mus, log_sigmas

    def log_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        logits, mus, log_sigmas = self.forward(s)
        # a: (B, D), mus: (B, K, D)
        a_expand = a.unsqueeze(-2)  # (B, 1, D)
        sigmas = log_sigmas.exp()

        # Per-component log-prob: sum over action dims
        log_comp = -0.5 * (
            self.action_dim * torch.log(torch.tensor(2 * torch.pi))
            + 2 * log_sigmas.sum(dim=-1)
            + ((a_expand - mus) ** 2 / (sigmas ** 2)).sum(dim=-1)
        )  # (B, K)

        # log-sum-exp with mixing weights
        log_weights = F.log_softmax(logits, dim=-1)  # (B, K)
        return torch.logsumexp(log_weights + log_comp, dim=-1)  # (B,)

    def sample(self, s: torch.Tensor, n: int = 1) -> torch.Tensor:
        logits, mus, log_sigmas = self.forward(s)
        sigmas = log_sigmas.exp()
        B = s.shape[0]

        if n == 1:
            # Choose components
            k = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)  # (B,)
            mu_k = mus[torch.arange(B), k]  # (B, D)
            sigma_k = sigmas[torch.arange(B), k]  # (B, D)
            return mu_k + sigma_k * torch.randn_like(mu_k)
        else:
            samples = []
            for _ in range(n):
                k = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
                mu_k = mus[torch.arange(B), k]
                sigma_k = sigmas[torch.arange(B), k]
                samples.append(mu_k + sigma_k * torch.randn_like(mu_k))
            return torch.stack(samples, dim=0)  # (n, B, D)

    def mean_action(self, s: torch.Tensor) -> torch.Tensor:
        """Weighted mean across all components."""
        logits, mus, log_sigmas = self.forward(s)
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)  # (B, K, 1)
        return (weights * mus).sum(dim=-2)  # (B, D)

    def component_info(self, s: torch.Tensor):
        """Return per-component weights, means, stds for visualization."""
        logits, mus, log_sigmas = self.forward(s)
        weights = F.softmax(logits, dim=-1)
        return weights, mus, log_sigmas.exp()
