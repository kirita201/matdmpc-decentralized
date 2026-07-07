import torch
import torch.nn as nn
import algorithm.helper as h
from algorithm.models import TransformerComm

class MFActor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_comm = getattr(cfg, "comm_type", "none") in ["local", "global"]
        
        self.enc = h.mlp(cfg.obs_shape[0], cfg.enc_dim, cfg.latent_dim)
        if self.use_comm:
            self.comm = TransformerComm(cfg)
        
        # 連続値 [0, 1] を出力するため Sigmoid を使用
        self.out = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)

    def forward(self, obs, adj_mask=None):
        e = self.enc(obs)
        if self.use_comm:
            # TransformerComm は行動(a)も要求するため、Actor の推論時は 0 テンソルでパディング
            dummy_a = torch.zeros(e.shape[0], self.cfg.num_agents, self.cfg.action_dim, device=e.device)
            e = self.comm(e, dummy_a, adj_mask)
        
        mu = torch.sigmoid(self.out(e))
        return mu

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