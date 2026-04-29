# envs/prey_policy.py
"""
Prey (good agent) の訓練済みポリシー。

設計方針
--------
* Prey は adversary (MA-TDMPC) とは独立に事前訓練しておく。
* ネットワークは軽量な MLP (obs -> action)。
* チェックポイントは dict 形式で保存/ロードする:
    {
        'obs_dim'   : int,
        'action_dim': int,
        'hidden_dim': int,
        'n_prey'    : int,
        'state_dict': { agent_id: OrderedDict, ... }  # 各 prey の重み
                       OR 共有重みの場合は単一 OrderedDict
    }

使い方
------
    from envs.prey_policy import PreyPolicy
    prey_policy = PreyPolicy.load("checkpoints/prey_N6.pt", device="cuda")
    actions = prey_policy.act(prey_obs)   # prey_obs: np.ndarray [n_prey, obs_dim]
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union


# ─────────────────────────────────────────────────────────────
# ネットワーク定義
# ─────────────────────────────────────────────────────────────

class PreyNet(nn.Module):
    """
    軽量 MLP ポリシー (共有重み or エージェント別重み)。
    出力は tanh で [-1, 1] にクリップ。
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (self.net(obs) + 1.0) / 2.0


# ─────────────────────────────────────────────────────────────
# ポリシーラッパー
# ─────────────────────────────────────────────────────────────

class PreyPolicy:
    """
    訓練済み Prey ポリシーのラッパー。

    Parameters
    ----------
    nets : list[PreyNet]
        n_prey 個のネットワーク (共有重みでも別重みでも良い)。
    obs_dim : int
    action_dim : int
    device : str
    noise_std : float
        推論時に加えるガウスノイズ (0 = 決定論的)。
        Prey を少しランダムに動かしたい場合に使う。
    """

    def __init__(
        self,
        nets: list,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
        noise_std: float = 0.0,
    ):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.noise_std = noise_std
        self.nets = [net.to(self.device).eval() for net in nets]

    # ── 推論 ────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, prey_obs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        prey_obs : np.ndarray, shape (n_prey, obs_dim)

        Returns
        -------
        actions : np.ndarray, shape (n_prey, action_dim)
        """
        actions = []
        for i, net in enumerate(self.nets):
            obs_t = torch.tensor(
                prey_obs[i], dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # [1, obs_dim]
            a = net(obs_t).squeeze(0)  # [action_dim]
            if self.noise_std > 0.0:
                a = a + self.noise_std * torch.randn_like(a)
                a = a.clamp(0, 1.0)
            actions.append(a.cpu().numpy())
        return np.stack(actions)  # [n_prey, action_dim]

    # ── 保存/ロード ─────────────────────────────────────────

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.nets[0].net[0].out_features,
            "n_prey": len(self.nets),
            "state_dicts": [net.state_dict() for net in self.nets],
        }
        torch.save(ckpt, path)
        print(f"[PreyPolicy] saved → {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: str = "cpu",
        noise_std: float = 0.0,
    ) -> "PreyPolicy":
        ckpt = torch.load(path, map_location=device)
        obs_dim = ckpt["obs_dim"]
        action_dim = ckpt["action_dim"]
        hidden_dim = ckpt.get("hidden_dim", 128)
        state_dicts = ckpt["state_dicts"]

        nets = []
        for sd in state_dicts:
            net = PreyNet(obs_dim, action_dim, hidden_dim)
            net.load_state_dict(sd)
            nets.append(net)

        policy = cls(nets, obs_dim, action_dim, device=device, noise_std=noise_std)
        print(
            f"[PreyPolicy] loaded from {path}  "
            f"(n_prey={len(nets)}, obs_dim={obs_dim}, action_dim={action_dim})"
        )
        return policy

    @classmethod
    def random(
        cls,
        n_prey: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
    ) -> "PreyPolicy":
        """
        チェックポイントが無い場合のフォールバック用ランダムポリシー。
        (重みは未学習 + ノイズ付きで実質ランダムに近い動きをする)
        """
        nets = [PreyNet(obs_dim, action_dim) for _ in range(n_prey)]
        return cls(nets, obs_dim, action_dim, device=device, noise_std=1.0)