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

        self._positions = torch.empty((self.capacity+1, self.N, 2), dtype=torch.float32, device=self.device)

    def add(self, obs, action, reward, done, info):
        self._obs[self.idx] = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self._action[self.idx] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self._reward[self.idx] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self._done[self.idx] = torch.tensor(done, dtype=torch.bool, device=self.device)

        if 'agent_positions' in info:
            self._positions[self.idx] = torch.tensor(info['agent_positions'], dtype=torch.float32, device=self.device)
        
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
        probs /= (probs.sum() + 1e-8)
        total = len(probs)
        
        if self._valid_mask_dirty:
            self._valid_mask = self._compute_valid_mask()
            self._valid_mask_dirty = False
        probs_filtered = probs * self._valid_mask.float()
        probs_filtered /= (probs_filtered.sum() + 1e-8)
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
        positions = self._positions[idx_mat]

        return obs, next_obs, action, reward, positions, idxs, weights
    
    def save(self, filepath):
        """バッファの状態をCPUメモリに移して保存"""
        state = {
            'obs': self._obs.cpu(),
            'action': self._action.cpu(),
            'reward': self._reward.cpu(),
            'done': self._done.cpu(),
            'priorities': self._priorities.cpu(),
            'idx': self.idx,
            'full': self._full
        }
        torch.save(state, filepath)

    def load(self, filepath):
        """バッファの状態をデバイスに読み込み（容量変更・リングバッファの再配置対応版）"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        old_capacity = checkpoint['obs'].shape[0] - 1
        old_idx = checkpoint['idx']
        old_full = checkpoint['full']
        
        # 容量が変わらない場合はそのままロード
        if old_capacity == self.capacity:
            self._obs.copy_(checkpoint['obs'])
            self._action.copy_(checkpoint['action'])
            self._reward.copy_(checkpoint['reward'])
            self._done.copy_(checkpoint['done'])
            self._priorities.copy_(checkpoint['priorities'])
            self.idx = old_idx
            self._full = old_full
            self._valid_mask_dirty = True
            print(f"[ReplayBuffer] Loaded buffer state from {filepath} (Same capacity)")
            return

        print(f"[ReplayBuffer] Adapting buffer capacity from {old_capacity} to {self.capacity}...")
        
        # 以前のバッファに格納されていた実際のデータ数
        old_stored_count = old_capacity if old_full else old_idx
        
        # 新しいバッファに引き継ぐデータ数
        copy_len = min(self.capacity, old_stored_count)
        
        # 有効なデータのインデックスを古い順（時系列順）に並べた配列を作成
        if old_full:
            # 満杯だった場合: idxから末尾までが古く、0からidx-1までが新しい
            indices = torch.arange(old_capacity, device=self.device)
            indices = (indices + old_idx) % old_capacity
        else:
            # 満杯でなかった場合: 0からidx-1までが有効
            indices = torch.arange(old_idx, device=self.device)
            
        # 新しい容量に収まりきらない場合は、最新のデータを優先して残す
        indices = indices[-copy_len:]
        
        # データを先頭(0)から順に詰め直す
        self._obs[:copy_len] = checkpoint['obs'][:-1][indices]
        self._action[:copy_len] = checkpoint['action'][indices]
        self._reward[:copy_len] = checkpoint['reward'][indices]
        self._done[:copy_len] = checkpoint['done'][indices]
        self._priorities[:copy_len] = checkpoint['priorities'][indices]
        
        # 元のバッファで最新だったデータの「次」の観測値（next_obs用）を末尾にセット
        next_idx = (indices[-1] + 1) % old_capacity
        self._obs[copy_len] = checkpoint['obs'][next_idx]
        
        # ★ご指摘の通り、新しい状態に合わせて正しくフラグとインデックスを更新
        self.idx = copy_len % self.capacity
        self._full = (copy_len == self.capacity) 
        self._valid_mask_dirty = True
        
        print(f"[ReplayBuffer] Successfully adapted capacity. New idx: {self.idx}, Full: {self._full}")