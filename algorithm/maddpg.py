import torch
import numpy as np
from copy import deepcopy
import algorithm.helper as h
from algorithm.model_mf import MFActor, MFCritic

class MADDPG:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.comm_range = float(getattr(cfg, 'comm_range', 'inf')) if cfg.comm_type == "local" else float('inf')
        
        self.actor = MFActor(cfg).to(self.device)
        self.critic = MFCritic(cfg, use_action=True).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.N = cfg.num_agents

    def _make_adj_mask(self, positions):
        if self.cfg.comm_type == "none" or self.comm_range == float('inf') or positions is None:
            return None
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
        dists = torch.norm(positions.unsqueeze(-2) - positions.unsqueeze(-3), dim=-1)
        return dists <= self.comm_range

    @torch.no_grad()
    def plan(self, obs, positions=None, eval_mode=False, step=None, t0=True):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        adj_mask = self._make_adj_mask(np.expand_dims(positions, axis=0) if positions is not None else None)
        
        a = self.actor(obs_t, adj_mask).squeeze(0)
        if not eval_mode:
            std = h.linear_schedule(self.cfg.std_schedule, step)
            noise = torch.randn_like(a) * std
            a = (a + noise).clamp(0.0, 1.0)
        return a

    def update(self, replay_buffer, step):
        beta = h.linear_schedule(self.cfg.per_beta, step)
        obs, next_obses, action, reward, positions, idxs, weights = replay_buffer.sample(beta)

        # MA-TDMPC のバッファ [H, B, N, dim] から先頭 1 ステップを抽出
        o = obs
        no = next_obses[0]
        a = action[0]
        r = reward[0]
        pos = positions[0]
        
        with torch.no_grad():
            adj_mask = self._make_adj_mask(pos)
            next_a = self.actor_target(no, adj_mask)
            next_q = self.critic_target(no, next_a)
            target_q = r + self.cfg.discount * next_q

        # Critic 更新
        current_q = self.critic(o, a)
        q_loss = (h.mse(current_q, target_q, reduce=False).mean(dim=1) * weights).mean()
        
        self.critic_optim.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip_norm)
        self.critic_optim.step()

        # Actor 更新 (CTDE)
        adj_mask_curr = self._make_adj_mask(positions[0])
        policy_a = self.actor(o, adj_mask_curr)
        actor_loss = -self.critic(o, policy_a).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
        self.actor_optim.step()

        # Target EMA
        if step % self.cfg.update_freq == 0:
            h.ema(self.actor, self.actor_target, self.cfg.tau)
            h.ema(self.critic, self.critic_target, self.cfg.tau)

        safe_p_loss = torch.nan_to_num(q_loss.detach().clamp(max=1e4).float(), nan=1.0)
        replay_buffer.update_priorities(idxs, safe_p_loss.expand_as(idxs))

        return {'q_loss': q_loss.detach(), 'actor_loss': actor_loss.detach()}

    def save(self, filepath):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])