# algorithm/buffer.py
import torch
import numpy as np
import torch.nn.functional as F

class ReplayBuffer:
    """Multi-Agent Prioritized Replay Buffer"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.capacity = min(cfg.train_steps, cfg.max_buffer_size)
        self.N = cfg.num_agents
        
        obs_shape = cfg.obs_shape 
        self._obs = torch.empty((self.capacity+1, self.N, *obs_shape), dtype=torch.float32, device=self.device)
        self._action = torch.empty((self.capacity, self.N, cfg.action_dim), dtype=torch.float32, device=self.device)
        self._reward = torch.empty((self.capacity, self.N), dtype=torch.float32, device=self.device)
        self._done = torch.empty((self.capacity, 1), dtype=torch.bool, device=self.device)
        
        self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
        self._eps = 1e-6
        self._full = False
        self.idx = 0

        self._valid_mask = None
        self._valid_mask_dirty = True

    def add(self, obs, action, reward, done):
        self._obs[self.idx] = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self._action[self.idx] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self._reward[self.idx] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self._done[self.idx] = torch.tensor(done, dtype=torch.bool, device=self.device)
        
        max_priority = self._priorities.max().item() if self.idx > 0 else 1.0
        self._priorities[self.idx] = max_priority
        
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self._full = True
        self._valid_mask_dirty = True

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.squeeze(-1).to(self.device) + self._eps

    def _compute_valid_mask(self):
        total = self.capacity if self._full else self.idx
        done_1d = self._done[:total].squeeze(-1).float()  # [T]

        any_done = F.max_pool1d(
            done_1d.unsqueeze(0).unsqueeze(0),
            kernel_size=self.cfg.horizon,
            stride=1,
            padding=0
        ).squeeze().bool()  # [T - horizon + 1]

        valid = torch.zeros(total, dtype=torch.bool, device=self.device)
        valid[:total - self.cfg.horizon + 1] = ~any_done
        return valid

    def sample(self, beta):
        probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
        probs /= probs.sum()
        total = len(probs)
        
        if self._valid_mask_dirty:
            self._valid_mask = self._compute_valid_mask()
            self._valid_mask_dirty = False
        probs_filtered = probs * self._valid_mask.float()
        probs_filtered /= probs_filtered.sum()
        idxs = torch.multinomial(probs_filtered, self.cfg.batch_size, replacement=True)
        weights = (total * probs[idxs]) ** (-beta)
        weights /= weights.max()

        obs = self._obs[idxs]
        
        next_obs = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, self.N, *self.cfg.obs_shape), dtype=torch.float32, device=self.device)
        action = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, self.N, self.cfg.action_dim), dtype=torch.float32, device=self.device)
        reward = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, self.N), dtype=torch.float32, device=self.device)
        
        """
        for t in range(self.cfg.horizon+1):
            _idxs = idxs + t
            next_obs[t] = self._obs[_idxs + 1]
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]
        """

        offsets = torch.arange(self.cfg.horizon + 1, device=self.device)  # [H+1]
        idx_mat = (idxs.unsqueeze(0) + offsets.unsqueeze(1))  # [H+1, B]

        next_obs = self._obs[idx_mat + 1]    # [H+1, B, N, obs_dim]
        action   = self._action[idx_mat]     # [H+1, B, N, action_dim]
        reward   = self._reward[idx_mat]     # [H+1, B, N]

        return obs, next_obs, action, reward, idxs, weights