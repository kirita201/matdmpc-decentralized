import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from algorithm.model_mf import MFActor, MFCritic

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.positions = []

    def store(self, obs, action, log_prob, reward, value, mask, pos):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.masks.append(mask)
        self.positions.append(pos)

class MAPPO:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.comm_range = float(getattr(cfg, 'comm_range', 'inf')) if cfg.comm_type == "local" else float('inf')
        
        self.actor = MFActor(cfg).to(self.device)
        self.critic = MFCritic(cfg, use_action=False).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        
        self.buffer = RolloutBuffer()
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
        
        mu = self.actor(obs_t, adj_mask)
        if eval_mode:
            return mu.squeeze(0)
            
        std = torch.ones_like(mu) * getattr(self.cfg, "min_std", 0.1)
        dist = Normal(mu, std)
        action = dist.sample()
        action_clipped = action.clamp(0.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs_t)
        
        return action_clipped.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def update(self, next_obs_tensor, next_pos_tensor, next_done):
        # テンソルへのスタック
        obs = torch.stack(self.buffer.obs)
        actions = torch.stack(self.buffer.actions)
        old_log_probs = torch.stack(self.buffer.log_probs)
        rewards = torch.stack(self.buffer.rewards)
        values = torch.stack(self.buffer.values)
        masks = torch.stack(self.buffer.masks)
        
        with torch.no_grad():
            next_mask = self._make_adj_mask(next_pos_tensor.unsqueeze(0) if next_pos_tensor is not None else None)
            next_value = self.critic(next_obs_tensor.unsqueeze(0))
            
        # GAEの計算
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value.squeeze(0)
            else:
                nextnonterminal = masks[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.cfg.discount * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.cfg.discount * self.cfg.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

        # ミニバッチ最適化
        b_obs, b_actions, b_old_log_probs, b_advantages, b_returns, b_pos = (
            obs.view(-1, self.N, self.cfg.obs_shape[0]),
            actions.view(-1, self.N, self.cfg.action_dim),
            old_log_probs.view(-1, self.N),
            advantages.view(-1, self.N),
            returns.view(-1, self.N),
            torch.stack(self.buffer.positions) if self.buffer.positions[0] is not None else [None]*len(obs)
        )

        total_a_loss, total_c_loss = 0, 0
        for _ in range(self.cfg.ppo_epochs):
            adj_mask = self._make_adj_mask(b_pos) if isinstance(b_pos, torch.Tensor) else None
            
            mu = self.actor(b_obs, adj_mask)
            std = torch.ones_like(mu) * getattr(self.cfg, "min_std", 0.1)
            dist = Normal(mu, std)
            new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
            entropy = dist.entropy().mean()
            
            new_values = self.critic(b_obs)
            
            ratio = torch.exp(new_log_probs - b_old_log_probs)
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip_param, 1.0 + self.cfg.ppo_clip_param) * b_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_coef * entropy
            critic_loss = F.mse_loss(new_values, b_returns)

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip_norm)
            self.critic_optim.step()
            
            total_a_loss += actor_loss.item()
            total_c_loss += critic_loss.item()

        self.buffer.clear()
        return {'actor_loss': total_a_loss / self.cfg.ppo_epochs, 'critic_loss': total_c_loss / self.cfg.ppo_epochs}

    def save(self, filepath):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])