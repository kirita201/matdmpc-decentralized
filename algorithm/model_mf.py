import torch
import torch.nn as nn

import algorithm.helper as h
from algorithm.models import TransformerComm


class MFActor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_comm = (
            getattr(cfg, "comm_type", "none") in ("local", "global")
        )

        self.enc = h.mlp(
            cfg.obs_shape[0], cfg.enc_dim, cfg.latent_dim
        )
        if self.use_comm:
            self.comm = TransformerComm(cfg)

        self.out = h.mlp(
            cfg.latent_dim, cfg.mlp_dim, cfg.action_dim
        )

    def forward(self, obs, adj_mask=None):
        e = self.enc(obs)

        if self.use_comm:
            dummy_a = e.new_zeros(
                *e.shape[:-1], self.cfg.action_dim
            )
            e = self.comm(e, dummy_a, adj_mask)

        return torch.sigmoid(self.out(e))


class MFCritic(nn.Module):
    def __init__(self, cfg, use_action=False):
        super().__init__()
        self.use_action = use_action
        self.N = cfg.num_agents

        in_dim = cfg.obs_shape[0] * self.N
        if use_action:
            in_dim += cfg.action_dim * self.N

        self.net = h.mlp(in_dim, cfg.mlp_dim, self.N)

    def forward(self, obs, action=None):
        batch_size = obs.shape[0]
        x = obs.reshape(batch_size, -1)

        if self.use_action:
            if action is None:
                raise ValueError(
                    "MFCritic(use_action=True) requires action."
                )
            x = torch.cat(
                [x, action.reshape(batch_size, -1)],
                dim=-1,
            )

        return self.net(x)