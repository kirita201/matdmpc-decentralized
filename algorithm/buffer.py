# algorithm/buffer.py
import torch
import numpy as np

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

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.squeeze(-1).to(self.device) + self._eps

    def sample(self, beta):
        probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
        probs /= probs.sum()
        total = len(probs)
        
        # Valid indices (ensure we can sample a full horizon)
        # --- 修正箇所 ---
        valid_idxs = []
        # CPU転送を避け、GPU上で一括サンプリング (多めに取得してフィルタリング)
        while len(valid_idxs) < self.cfg.batch_size:
            sampled_idxs = torch.multinomial(probs, int(self.cfg.batch_size * 1.5), replacement=True).tolist()
            for idx in sampled_idxs:
                if (idx + self.cfg.horizon < total) and not self._done[idx:idx+self.cfg.horizon].any():
                    valid_idxs.append(idx)
                    if len(valid_idxs) == self.cfg.batch_size:
                        break
                    
        idxs = torch.tensor(valid_idxs, device=self.device, dtype=torch.long)
        weights = (total * probs[idxs]) ** (-beta)
        weights /= weights.max()

        obs = self._obs[idxs]
        
        next_obs = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, self.N, *self.cfg.obs_shape), dtype=torch.float32, device=self.device)
        action = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, self.N, self.cfg.action_dim), dtype=torch.float32, device=self.device)
        reward = torch.empty((self.cfg.horizon+1, self.cfg.batch_size, self.N), dtype=torch.float32, device=self.device)
        
        for t in range(self.cfg.horizon+1):
            _idxs = idxs + t
            next_obs[t] = self._obs[_idxs + 1]
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]

        return obs, next_obs, action, reward, idxs, weights