# algorithm/model.py
import torch
import torch.nn as nn
import algorithm.helper as h

class TransformerComm(nn.Module):
    """Transformer-based Communication Module"""
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.latent_dim + cfg.action_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=cfg.n_heads, 
            dim_feedforward=cfg.mlp_dim, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_com_layers)
        self.out_proj = nn.Linear(input_dim, cfg.latent_dim)

    def forward(self, e, a):
        # e: [Batch, N, latent_dim], a: [Batch, N, action_dim]
        x = torch.cat([e, a], dim=-1) # [Batch, N, latent_dim + action_dim]
        z = self.transformer(x)
        return self.out_proj(z)

class MixingNetwork(nn.Module):
    """Value Decomposition Network (QMIX-style but without monotonicity constraint)"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.num_agents
        state_dim = cfg.latent_dim * self.N
        
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, cfg.mlp_dim), nn.ELU(),
            nn.Linear(cfg.mlp_dim, self.N * cfg.mlp_dim)
        )
        self.hyper_b1 = nn.Linear(state_dim, cfg.mlp_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, cfg.mlp_dim), nn.ELU(),
            nn.Linear(cfg.mlp_dim, cfg.mlp_dim * 1)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, cfg.mlp_dim), nn.ELU(),
            nn.Linear(cfg.mlp_dim, 1)
        )

    def forward(self, q_values, global_state):
        # q_values: [B, N]
        # global_state: [B, N * latent_dim]
        B = q_values.size(0)
        q_values = q_values.view(B, 1, self.N)
        
        w1 = self.hyper_w1(global_state).view(B, self.N, self.cfg.mlp_dim)
        b1 = self.hyper_b1(global_state).view(B, 1, self.cfg.mlp_dim)
        
        hidden = nn.functional.elu(torch.bmm(q_values, w1) + b1)
        
        w2 = self.hyper_w2(global_state).view(B, self.cfg.mlp_dim, 1)
        b2 = self.hyper_b2(global_state).view(B, 1, 1)
        
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(B, 1)

class MACLM(nn.Module):
    """Multi-Agent Communication-based Local Model"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.num_agents
        
        # 共通モジュール (全エージェントでパラメータ共有)
        self._encoder = h.mlp(cfg.obs_shape[0], cfg.enc_dim, cfg.latent_dim)
        self._comm = TransformerComm(cfg)
        
        self._dynamics = h.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
        self._reward = h.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, 1)
        
        self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
        
        # Q-networks (Double Q-learning)
        self._Q1 = h.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, 1)
        self._Q2 = h.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, 1)
        
        self._mixing1 = MixingNetwork(cfg)
        self._mixing2 = MixingNetwork(cfg)

        self.apply(h.orthogonal_init)

    def track_q_grad(self, enable=True):
        for m in [self._Q1, self._Q2, self._mixing1, self._mixing2]:
            h.set_requires_grad(m, enable)

    def encode(self, obs):
        """obs: [B, N, obs_dim] -> e: [B, N, latent_dim]"""
        return self._encoder(obs)

    def communicate(self, e, a):
        return self._comm(e, a)

    def next(self, z, a):
        """z: [B, N, latent_dim], a: [B, N, action_dim]"""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x).squeeze(-1) # return [B, N]

    def pi(self, z, std=0):
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std_t = torch.ones_like(mu) * std
            return h.TruncatedNormal(mu, std_t).sample(clip=0.3)
        return mu

    def Q(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self._Q1(x).squeeze(-1), self._Q2(x).squeeze(-1) # return [B, N]
    
    def Q_joint(self, qs1, qs2, e):
        """qs: [B, N], e: [B, N, latent_dim]"""
        B = e.size(0)
        global_state = e.view(B, -1)
        return self._mixing1(qs1, global_state), self._mixing2(qs2, global_state)