from copy import deepcopy

import numpy as np
import torch

import algorithm.helper as h
from algorithm.model_mf import MFActor, MFCritic


class MADDPG:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.comm_range = (
            float(getattr(cfg, "comm_range", "inf"))
            if cfg.comm_type == "local"
            else float("inf")
        )

        self.actor = MFActor(cfg).to(self.device)
        self.critic = MFCritic(cfg, use_action=True).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.lr
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.lr
        )
        self.N = cfg.num_agents

    def _make_adj_mask(self, positions):
        if (
            self.cfg.comm_type == "none"
            or self.comm_range == float("inf")
            or positions is None
        ):
            return None

        positions = torch.as_tensor(
            positions, dtype=torch.float32, device=self.device
        )
        dists = torch.norm(
            positions.unsqueeze(-2) - positions.unsqueeze(-3),
            dim=-1,
        )
        return dists <= self.comm_range

    @torch.no_grad()
    def plan(
        self, obs, positions=None, eval_mode=False, step=None, t0=True
    ):
        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        pos_t = (
            np.expand_dims(positions, axis=0)
            if positions is not None
            else None
        )
        adj_mask = self._make_adj_mask(pos_t)

        action = self.actor(obs_t, adj_mask).squeeze(0)

        if not eval_mode:
            std = h.linear_schedule(self.cfg.std_schedule, step)
            action = (
                action + torch.randn_like(action) * std
            ).clamp(0.0, 1.0)

        return action

    def update(self, replay_buffer, step):
        beta = h.linear_schedule(self.cfg.per_beta, step)
        obs, next_obses, actions, rewards, positions, idxs, weights = (
            replay_buffer.sample(beta)
        )

        # ReplayBuffer が返す時刻軸の先頭１ステップを使用。
        o = obs
        no = next_obses[0]
        a = actions[0]
        r = rewards[0]
        pos = positions[0]

        with torch.no_grad():
            # ご指定により、ここでは現在位置のマスクを使う。
            adj_mask = self._make_adj_mask(pos)
            next_a = self.actor_target(no, adj_mask)
            next_q = self.critic_target(no, next_a)

            # 個別報酬 [B, N]、共通報酬 [B] または [B, 1] に対応。
            r = r.reshape(next_q.shape[0], -1)
            if r.shape[1] not in (1, self.N):
                raise ValueError(
                    f"Unexpected reward shape: {tuple(r.shape)}"
                )

            # 真の終端はなく、時間切れ後もブートストラップする。
            target_q = r + self.cfg.discount * next_q

        current_q = self.critic(o, a)
        td_error = current_q - target_q

        # 各リプレイサンプルにつき１つの損失・優先度を計算。
        per_sample_loss = td_error.square().mean(dim=-1)
        weights = torch.as_tensor(
            weights, dtype=per_sample_loss.dtype, device=self.device
        ).reshape(-1)

        if weights.numel() != per_sample_loss.numel():
            raise ValueError(
                "PER weights must contain one value per sampled transition."
            )

        q_loss = (per_sample_loss * weights).mean()

        self.critic_optim.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.grad_clip_norm
        )
        self.critic_optim.step()

        adj_mask_curr = self._make_adj_mask(pos)
        policy_a = self.actor(o, adj_mask_curr)
        actor_loss = -self.critic(o, policy_a).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.cfg.grad_clip_norm
        )
        self.actor_optim.step()

        if step % self.cfg.update_freq == 0:
            h.ema(self.actor, self.actor_target, self.cfg.tau)
            h.ema(self.critic, self.critic_target, self.cfg.tau)

        priorities = (
            td_error.detach()
            .abs()
            .mean(dim=-1)
            .nan_to_num(nan=1.0, posinf=1e4, neginf=1.0)
            .clamp(min=1e-6, max=1e4)
        )

        if isinstance(idxs, torch.Tensor):
            priorities = priorities.reshape_as(idxs)
        else:
            priorities = priorities.cpu().numpy().reshape(
                np.asarray(idxs).shape
            )

        replay_buffer.update_priorities(idxs, priorities)

        return {
            "q_loss": q_loss.detach(),
            "actor_loss": actor_loss.detach(),
        }

    def save(self, filepath):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
            },
            filepath,
        )

    def load(self, filepath):
        checkpoint = torch.load(
            filepath, map_location=self.device, weights_only=False
        )

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

        # 旧形式のチェックポイントでは現在ネットワークから初期化。
        self.actor_target.load_state_dict(
            checkpoint.get("actor_target", checkpoint["actor"])
        )
        self.critic_target.load_state_dict(
            checkpoint.get("critic_target", checkpoint["critic"])
        )

        if "actor_optim" in checkpoint:
            self.actor_optim.load_state_dict(checkpoint["actor_optim"])
        if "critic_optim" in checkpoint:
            self.critic_optim.load_state_dict(checkpoint["critic_optim"])