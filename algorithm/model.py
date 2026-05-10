# algorithm/model.py
import torch
import torch.nn as nn
import algorithm.helper as h

class TransformerComm(nn.Module):
    """Transformer-based Communication Module"""
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.latent_dim 
        
        # 入力を d_model に変換する層を追加
        self.in_proj = nn.Linear(cfg.latent_dim + cfg.action_dim, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=cfg.n_heads, 
            dim_feedforward=cfg.mlp_dim, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_com_layers)
        self.out_proj = nn.Linear(self.d_model, cfg.latent_dim)

    def forward(self, e, a, adj_mask=None):
        """
        Parameters
        ----------
        e        : Tensor [B, N, latent_dim]
        a        : Tensor [B, N, action_dim]
        adj_mask : BoolTensor [B, N, N] または None
            True の位置は「通信可能（アテンション許可）」。
            None の場合は全結合（従来動作と等価）。

        Returns
        -------
        Tensor [B, N, latent_dim]

        Notes
        -----
        nn.TransformerEncoder の src_key_padding_mask / attn_mask は
        「True = 無視（マスクアウト）」の符号規約を使う。
        一方 adj_mask は「True = 通信可能」なので論理反転して渡す。

        TransformerEncoderLayer に渡す attn_mask は
        shape [B*n_heads, N, N] か [N, N] のどちらかを受け付ける。
        バッチごとに異なるマスクを使いたいので [B*n_heads, N, N] で展開する。
        """
        B, N, _ = e.shape
        x = torch.cat([e, a], dim=-1)   # [B, N, latent_dim + action_dim]
        x = self.in_proj(x)             # [B, N, d_model]

        src_mask = None
        if adj_mask is not None:
            # adj_mask: [B, N, N], True = 通信可能
            # PyTorch の attn_mask 規約: -inf で遮断、0.0 で通過
            # n_heads 分に複製: [B*n_heads, N, N]
            n_heads = self.transformer.layers[0].self_attn.num_heads
            # [B, N, N] -> [B, 1, N, N] -> [B, n_heads, N, N] -> [B*n_heads, N, N]
            mask_expanded = adj_mask.unsqueeze(1).expand(B, n_heads, N, N)
            mask_expanded = mask_expanded.reshape(B * n_heads, N, N)
            # True(通信可) → 0.0、False(遮断) → -inf
            src_mask = torch.zeros_like(mask_expanded, dtype=x.dtype)
            src_mask = src_mask.masked_fill(~mask_expanded, float('-inf'))

        z = self.transformer(x, mask=src_mask)
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

    def communicate(self, e, a, adj_mask=None):
        """
        Parameters
        ----------
        e        : Tensor [B, N, latent_dim]
        a        : Tensor [B, N, action_dim]
        adj_mask : BoolTensor [B, N, N] or None
            True = 通信可能。None のとき全結合（ベースラインと等価）。
        """
        return self._comm(e, a, adj_mask=adj_mask)

    def communicate_per_agent(self, e_per_agent, a, adj_mask_per_agent=None):
        """
        エージェントiごとに独立した通信グラフでアテンションを実行する。
        update() の [N, B, N, latent] テンソルをそのまま処理できるよう、
        N*B をバッチ次元に畳み込んで TransformerComm に通す。

        Parameters
        ----------
        e_per_agent        : Tensor [N, B, N, latent_dim]
            e_per_agent[i] = エージェントiの視点での全エージェント埋め込み [B, N, latent]
        a                  : Tensor [B, N, action_dim]
            全エージェントの行動（全視点で共通）
        adj_mask_per_agent : BoolTensor [N, B, N, N] or None
            adj_mask_per_agent[i, b] = エージェントiの視点でのサンプルbの通信グラフ [N, N]

        Returns
        -------
        z_per_agent : Tensor [N, B, N, latent_dim]
            z_per_agent[i, b, j] = エージェントiの通信後のエージェントj埋め込み
        """
        N, B, _, latent = e_per_agent.shape

        # [N, B, N, latent] -> [N*B, N, latent]
        e_flat = e_per_agent.reshape(N * B, N, latent)

        # a: [B, N, action_dim] -> [N*B, N, action_dim]
        a_flat = a.unsqueeze(0).expand(N, B, N, -1).reshape(N * B, N, -1)

        # マスクを [N*B, N, N] に畳む
        mask_flat = adj_mask_per_agent.reshape(N * B, N, N) if adj_mask_per_agent is not None else None

        z_flat = self._comm(e_flat, a_flat, adj_mask=mask_flat)  # [N*B, N, latent]

        return z_flat.reshape(N, B, N, latent)

    def next(self, z, a):
        """z: [B, N, latent_dim], a: [B, N, action_dim]"""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x).squeeze(-1) # return [B, N]

    def pi(self, z, std=0):
        mu = (torch.tanh(self._pi(z)) + 1.0) / 2.0
        if std > 0:
            std_t = torch.ones_like(mu) * std
            return h.TruncatedNormal(mu, std_t, low=0.0, high=1.0).sample(clip=0.3)
        return mu

    def Q(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self._Q1(x).squeeze(-1), self._Q2(x).squeeze(-1) # return [B, N]
    
    def Q_joint(self, qs1, qs2, e):
        """qs: [B, N], e: [B, N, latent_dim]"""
        B = e.size(0)
        global_state = e.view(B, -1)
        return self._mixing1(qs1, global_state), self._mixing2(qs2, global_state)