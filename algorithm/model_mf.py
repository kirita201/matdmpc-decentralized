import torch
import torch.nn as nn
import algorithm.helper as h
from algorithm.models import TransformerComm

class MFActor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_comm = getattr(
            cfg,
            "comm_type",
            "none",
        ) in ("local", "global")

        self.enc = h.mlp(
            cfg.obs_shape[0],
            cfg.enc_dim,
            cfg.latent_dim,
        )

        if self.use_comm:
            # MAPPO の行動時と PPO 更新時で dropout による
            # 方策の確率変動を起こさない。
            self.comm = TransformerComm(cfg, dropout=0.0)

        self.out = h.mlp(
            cfg.latent_dim,
            cfg.mlp_dim,
            cfg.action_dim,
        )

    def forward(self, obs, adj_mask=None):
        e = self.enc(obs)

        if self.use_comm:
            dummy_a = torch.zeros(
                *e.shape[:-1],
                self.cfg.action_dim,
                device=e.device,
                dtype=e.dtype,
            )
            e = self.comm(e, dummy_a, adj_mask)

        # sigmoid はここで掛けない。
        # Normal 分布の「変換前の平均」を返す。
        return self.out(e)

class MFCritic(nn.Module):
    def __init__(self, cfg, use_action=False):
        super().__init__()
        self.use_action = use_action
        self.N = cfg.num_agents
        
        # CTDE: 全エージェントの観測（＋行動）を平坦化して入力
        in_dim = cfg.obs_shape[0] * self.N
        if use_action:
            in_dim += cfg.action_dim * self.N
            
        self.net = h.mlp(in_dim, cfg.mlp_dim, self.N)

    def forward(self, obs, action=None):
        B = obs.shape[0]
        x = obs.view(B, -1)
        if self.use_action and action is not None:
            x = torch.cat([x, action.view(B, -1)], dim=-1)
        return self.net(x)