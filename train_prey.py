#!/usr/bin/env python3
"""
train_prey.py
=============
Predator-Prey タスクにおける Prey (good agent) を
Adversary (捕食者) の動きに依存せず事前に訓練するスクリプト。

訓練設定
--------
* Prey は「捕食者から遠ざかる」という報酬で REINFORCE / Actor-Critic で学習。
* Adversary は **ランダム行動** で固定 (Prey のみ学習させる)。
* 学習後のチェックポイントを `checkpoints/prey_N{N}.pt` に保存する。
* 保存されたチェックポイントは MPEWrapper が自動でロードして使用する。

使い方
------
    python train_prey.py --N 6 --steps 200000 --device cuda

引数
----
--N       : Adversary 数 (= num_agents)。3 / 6 / 15 から選ぶ
--steps   : 学習ステップ数
--device  : cuda / cpu
--ckpt_dir: チェックポイント保存先 (default: checkpoints/)
--lr      : 学習率
--hidden  : 隠れ層次元
--noise   : 推論時のノイズ (0 = 決定論的)
--seed    : 乱数シード
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# ── パス設定 (プロジェクトルートから実行することを想定) ────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from envs.mpe_wrapper import MPEWrapper, PREDPREY_CONFIGS
from envs.prey_policy import PreyNet, PreyPolicy


# ─────────────────────────────────────────────────────────────
# 値関数ネットワーク (Actor-Critic 用)
# ─────────────────────────────────────────────────────────────

class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ─────────────────────────────────────────────────────────────
# Prey 用簡易 Actor-Critic エージェント
# ─────────────────────────────────────────────────────────────

class PreyACAgent:
    """
    全 Prey エージェントでパラメータを共有する
    簡易 Actor-Critic エージェント。

    * Actor  : PreyNet (決定論的 + ガウスノイズ探索)
    * Critic : ValueNet (状態価値 V(s) の推定)
    * 更新   : n-step return + advantage による policy gradient
    """

    def __init__(self, obs_dim, action_dim, n_prey, hidden_dim=128, lr=3e-4, device="cpu"):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_prey = n_prey

        # 単一の Actor と Critic に変更
        self.actor  = PreyNet(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic = ValueNet(obs_dim, hidden_dim).to(self.device)

        self.actor_optim  = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # 探索用の標準偏差も全エージェントで共通の1つのパラメータとする
        self.log_std = nn.Parameter(
            torch.zeros(1, action_dim, device=self.device)
        )
        self.std_optim = torch.optim.Adam([self.log_std], lr=lr)

        # オンポリシーのトランジションバッファ
        self._reset_buffer()

    def _reset_buffer(self):
        self.buf_obs    = [[] for _ in range(self.n_prey)]
        self.buf_acts   = [[] for _ in range(self.n_prey)]
        self.buf_logp   = [[] for _ in range(self.n_prey)]
        self.buf_rews   = [[] for _ in range(self.n_prey)]
        self.buf_vals   = [[] for _ in range(self.n_prey)]
        self.buf_dones  = []

    @torch.no_grad()
    def act(self, prey_obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        prey_obs: (n_prey, obs_dim)
        returns : (n_prey, action_dim)
        """
        # 全エージェントの観測を一括でバッチ処理
        obs_t = torch.tensor(prey_obs, dtype=torch.float32, device=self.device)
        mean  = self.actor(obs_t)  # [n_prey, action_dim]
        
        if explore:
            std = self.log_std.exp().clamp(0.01, 1.0) # [1, action_dim] (ブロードキャストされる)
            dist = torch.distributions.Normal(mean, std)
            a = dist.sample()
        else:
            a = mean
            
        return a.clamp(-1, 1).cpu().numpy()

    @torch.no_grad()
    def store(self, prey_obs, actions, rewards, dones):
        """オンポリシーバッファにトランジションを追加。"""
        obs_t = torch.tensor(prey_obs, dtype=torch.float32, device=self.device)
        mean  = self.actor(obs_t)
        std   = self.log_std.exp().clamp(0.01, 1.0)
        dist  = torch.distributions.Normal(mean, std)
        a_t   = torch.tensor(actions, dtype=torch.float32, device=self.device)
        
        logp  = dist.log_prob(a_t).sum(dim=-1) # [n_prey]
        val   = self.critic(obs_t).squeeze(-1) # [n_prey]

        for i in range(self.n_prey):
            self.buf_obs[i].append(prey_obs[i])
            self.buf_acts[i].append(actions[i])
            self.buf_logp[i].append(logp[i].item())
            self.buf_rews[i].append(rewards[i])
            self.buf_vals[i].append(val[i].item())
        self.buf_dones.append(float(dones))

    def update(self, gamma: float = 0.99, lam: float = 0.95, entropy_coef: float = 0.01):
        """
        GAE (Generalized Advantage Estimation) + PPO-clip ライクな更新。
        全エージェントの経験を結合して1つのバッチとして学習する。
        """
        T = len(self.buf_dones)
        if T < 2:
            self._reset_buffer()
            return 0.0, 0.0

        all_obs   = []
        all_acts  = []
        all_advs  = []
        all_rets  = []
        all_logps = []

        done_arr = np.array(self.buf_dones, dtype=np.float32)

        # エージェントごとに時間方向の GAE を計算
        for i in range(self.n_prey):
            rew_arr  = np.array(self.buf_rews[i], dtype=np.float32)
            val_arr  = np.array(self.buf_vals[i], dtype=np.float32)

            adv  = np.zeros(T, dtype=np.float32)
            gae  = 0.0
            for t in reversed(range(T - 1)):
                delta = rew_arr[t] + gamma * val_arr[t + 1] * (1 - done_arr[t]) - val_arr[t]
                gae   = delta + gamma * lam * (1 - done_arr[t]) * gae
                adv[t] = gae
            returns = adv + val_arr

            all_obs.append(np.array(self.buf_obs[i], dtype=np.float32))
            all_acts.append(np.array(self.buf_acts[i], dtype=np.float32))
            all_advs.append(adv)
            all_rets.append(returns)
            all_logps.append(np.array(self.buf_logp[i], dtype=np.float32))

        # 全エージェントのデータをバッチ方向に結合 (Shape: [n_prey * T, ...])
        obs_t      = torch.tensor(np.concatenate(all_obs),   device=self.device)
        act_t      = torch.tensor(np.concatenate(all_acts),  device=self.device)
        adv_t      = torch.tensor(np.concatenate(all_advs),  device=self.device)
        ret_t      = torch.tensor(np.concatenate(all_rets),  device=self.device)
        logp_old_t = torch.tensor(np.concatenate(all_logps), device=self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ── Actor 更新 (全エージェントのデータで一括更新) ──
        self.actor_optim.zero_grad()
        self.std_optim.zero_grad()
        
        mean  = self.actor(obs_t)
        std   = self.log_std.exp().clamp(0.01, 1.0)
        dist  = torch.distributions.Normal(mean, std)
        logp  = dist.log_prob(act_t).sum(dim=-1)
        
        ratio = (logp - logp_old_t).exp()
        clip_ratio = ratio.clamp(1 - 0.2, 1 + 0.2)
        actor_loss = -torch.min(ratio * adv_t, clip_ratio * adv_t).mean()
        entropy_loss = -entropy_coef * dist.entropy().mean()
        
        (actor_loss + entropy_loss).backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()
        self.std_optim.step()

        # ── Critic 更新 (全エージェントのデータで一括更新) ──
        self.critic_optim.zero_grad()
        val_pred = self.critic(obs_t).squeeze()
        critic_loss = F.mse_loss(val_pred, ret_t)
        
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()

        self._reset_buffer()
        return actor_loss.item(), critic_loss.item()

    def get_policy(self, device: str = "cpu", noise_std: float = 0.0) -> PreyPolicy:
        """
        学習済み重みから PreyPolicy オブジェクトを生成して返す。
        同一のモデルへの参照を n_prey 個リストに入れて渡すことで、
        PreyPolicy の要件 (エージェント数分のリスト) を満たす。
        """
        return PreyPolicy(
            nets=[self.actor] * self.n_prey,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=device,
            noise_std=noise_std,
        )


# ─────────────────────────────────────────────────────────────
# 環境ラッパー: Prey の観測・報酬を個別に取得するヘルパー
# ─────────────────────────────────────────────────────────────

class PreyTrainEnv:
    """
    Prey 訓練専用の薄いラッパー。
    - Adversary はランダム行動で固定
    - Prey の観測・報酬・done を返す
    """

    def __init__(self, N: int, episode_length: int = 25):
        # 最小限の cfg オブジェクトを作る
        class _Cfg:
            pass
        cfg = _Cfg()
        cfg.task          = "predator_prey"
        cfg.num_agents    = N
        cfg.episode_length = episode_length

        from envs.mpe_wrapper import MPEWrapper, PREDPREY_CONFIGS
        self.env = MPEWrapper(cfg)

        # Prey 数・観測次元・行動次元を取得
        num_good = PREDPREY_CONFIGS[N][0]
        self.n_prey = num_good

        # Prey の観測次元は good_agent の観測次元
        # PettingZoo では good_agent も observe できる
        self._all_agents = self.env._all_agents
        self._prey_agents = [a for a in self._all_agents if "adversary" not in a]
        self._adv_agents  = [a for a in self._all_agents if "adversary" in a]

        # good agent の obs_dim を取得
        self.env.env.reset()
        prey_obs_sample = self.env.env.observe(self._prey_agents[0])
        self.obs_dim    = prey_obs_sample.shape[0]
        self.action_dim = self.env.env.action_space(self._prey_agents[0]).shape[0]

        print(
            f"[PreyTrainEnv] N_adv={N}, n_prey={num_good}, "
            f"prey_obs_dim={self.obs_dim}, action_dim={self.action_dim}"
        )

    def reset(self):
        self.env.env.reset()
        return self._get_prey_obs()

    def step(self, prey_actions: np.ndarray):
        """
        prey_actions: (n_prey, action_dim)
        adversary は act_space.sample() で固定
        """
        adv_idx  = 0
        prey_idx = 0
        for agent in self._all_agents:
            if "adversary" in agent:
                act = self.env.env.action_space(agent).sample()
            else:
                act = prey_actions[prey_idx]
                prey_idx += 1
            self.env.env.step(act)

        prey_obs     = self._get_prey_obs()
        prey_rewards = np.array(
            [self.env.env.rewards[a] for a in self._prey_agents], dtype=np.float32
        )
        dones = [
            self.env.env.terminations[a] or self.env.env.truncations[a]
            for a in self._prey_agents
        ]
        done = bool(np.any(dones))
        return prey_obs, prey_rewards, done, {}

    def _get_prey_obs(self):
        return np.stack([self.env.env.observe(a) for a in self._prey_agents])


# ─────────────────────────────────────────────────────────────
# メイン: 訓練ループ
# ─────────────────────────────────────────────────────────────

def train_prey(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    env = PreyTrainEnv(N=args.N, episode_length=args.episode_length)
    agent = PreyACAgent(
        obs_dim    = env.obs_dim,
        action_dim = env.action_dim,
        n_prey     = env.n_prey,
        hidden_dim = args.hidden,
        lr         = args.lr,
        device     = args.device,
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = Path(f"logs/prey_N{args.N}")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    total_steps = 0
    episode_idx = 0
    update_every = args.episode_length  # 1 エピソードごとに更新

    print(f"[train_prey] Start training Prey (N_adv={args.N}, steps={args.steps})")

    while total_steps < args.steps:
        obs  = env.reset()
        done = False
        ep_reward = 0.0
        t = 0

        while not done:
            actions = agent.act(obs, explore=True)
            next_obs, rewards, done, _ = env.step(actions)
            agent.store(obs, actions, rewards, done)
            obs = next_obs
            ep_reward += float(rewards.mean())
            t += 1
            total_steps += 1

        # エピソード終了 → 更新
        a_loss, c_loss = agent.update()

        writer.add_scalar("Prey/EpisodeReward", ep_reward, total_steps)
        writer.add_scalar("Prey/ActorLoss",     a_loss,    total_steps)
        writer.add_scalar("Prey/CriticLoss",    c_loss,    total_steps)

        episode_idx += 1
        if episode_idx % 200 == 0:
            print(
                f"  Step={total_steps:7d} | Ep={episode_idx:5d} | "
                f"Reward={ep_reward:.3f} | A_loss={a_loss:.4f} | C_loss={c_loss:.4f}"
            )

        # チェックポイント保存
        if episode_idx % args.save_every == 0 or total_steps >= args.steps:
            ckpt_path = ckpt_dir / f"prey_N{args.N}.pt"
            policy = agent.get_policy(device="cpu")
            policy.save(ckpt_path)

    # 最終保存
    ckpt_path = ckpt_dir / f"prey_N{args.N}.pt"
    policy = agent.get_policy(device="cpu")
    policy.save(ckpt_path)
    print(f"[train_prey] Done. Final checkpoint saved to {ckpt_path}")
    writer.close()


# ─────────────────────────────────────────────────────────────
# エントリポイント
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Prey policy for Predator-Prey")
    parser.add_argument("--N",            type=int,   default=6,
                        help="Number of adversaries (3 / 6 / 15)")
    parser.add_argument("--steps",        type=int,   default=300_000,
                        help="Total training steps")
    parser.add_argument("--episode_length", type=int, default=25)
    parser.add_argument("--device",       type=str,   default="cuda",
                        help="cuda or cpu")
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--hidden",       type=int,   default=128)
    parser.add_argument("--noise",        type=float, default=0.0,
                        help="Noise std at inference time (0=deterministic)")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--ckpt_dir",     type=str,   default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every",   type=int,   default=500,
                        help="Save checkpoint every N episodes")
    args = parser.parse_args()
    train_prey(args)