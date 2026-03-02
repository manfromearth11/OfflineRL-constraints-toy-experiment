import numpy as np
import ot as pot
import torch

from src.env.bandit_env import ground_truth_q, behavior_policy_sample
from src.methods.base import OfflineRLMethod
from src.models.policy_net import GaussianPolicy


class UnbalancedOTPolicy(OfflineRLMethod):
    """Unbalanced OT: soft marginal constraints via KL penalties."""

    def __init__(self, policy: GaussianPolicy, config: dict, env_config: dict, q_fn=None):
        super().__init__(policy, config)
        self.lam = config.get("lam", 1.0)
        self.entropic_reg = config.get("entropic_reg", 0.1)
        self.marginal_reg = config.get("marginal_reg", 1.0)
        self.n_ot_samples = config.get("n_ot_samples", 32)
        self.n_ot_states = config.get("n_ot_states", 16)
        self.env_config = env_config
        self.q_fn = q_fn or ground_truth_q

    @property
    def name(self) -> str:
        return "Unbalanced OT"

    def compute_loss(self, batch: dict) -> torch.Tensor:
        s = batch["states"]
        device = s.device
        B = s.shape[0]
        K = self.n_ot_samples

        a_full = self.policy.sample(s)
        q_vals = self.q_fn(s, a_full).mean()

        B_ot = min(self.n_ot_states, B)
        idx = torch.randperm(B)[:B_ot]
        s_sub = s[idx]

        s_expand = s_sub.unsqueeze(1).expand(B_ot, K, 2).reshape(B_ot * K, 2)
        a_policy = self.policy.sample(s_expand).reshape(B_ot, K, 2)

        s_np = s_sub.detach().cpu().numpy()
        a_behavior_all = []
        for i in range(B_ot):
            si = np.tile(s_np[i], (K, 1))
            a_behavior_all.append(behavior_policy_sample(si, self.env_config))
        a_behavior = torch.tensor(np.array(a_behavior_all), dtype=torch.float32, device=device)

        uniform = np.ones(K, dtype=np.float64) / K
        uot_loss = torch.tensor(0.0, device=device)
        for i in range(B_ot):
            C = torch.cdist(a_policy[i], a_behavior[i], p=2) ** 2
            C_np = C.detach().cpu().numpy().astype(np.float64)
            C_max = C_np.max()
            if C_max > 0:
                C_np_norm = C_np / C_max
            else:
                C_np_norm = C_np
            T_unbal = pot.unbalanced.sinkhorn_unbalanced(
                uniform, uniform, C_np_norm,
                reg=self.entropic_reg,
                reg_m=self.marginal_reg,
                numItermax=100,
            )
            T_unbal = np.nan_to_num(T_unbal, nan=0.0, posinf=0.0, neginf=0.0)
            T_tensor = torch.tensor(T_unbal, dtype=torch.float32, device=device)
            uot_loss = uot_loss + (T_tensor * C).sum()
        uot_loss = uot_loss / B_ot

        return -q_vals + self.lam * uot_loss
