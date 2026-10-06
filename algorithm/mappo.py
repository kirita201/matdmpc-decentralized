import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import (
    Independent,
    Normal,
    SigmoidTransform,
    TransformedDistribution,
)

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

    def store(
        self,
        obs,
        action,
        log_prob,
        reward,
        value,
        mask,
        pos,
    ):
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
        self.N = cfg.num_agents

        self.comm_range = (
            float(getattr(cfg, "comm_range", "inf"))
            if cfg.comm_type == "local"
            else float("inf")
        )

        self.actor = MFActor(cfg).to(self.device)
        self.critic = MFCritic(
            cfg,
            use_action=False,
        ).to(self.device)

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(),
            lr=cfg.lr,
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(),
            lr=cfg.lr,
        )

        self.buffer = RolloutBuffer()

    def _make_adj_mask(self, positions):
        if self.cfg.comm_type != "local":
            return None

        if positions is None:
            raise ValueError(
                "comm_type='local' ですが、"
                "agent_positions が取得できません。"
            )

        positions = torch.as_tensor(
            positions,
            dtype=torch.float32,
            device=self.device,
        )

        dists = torch.norm(
            positions.unsqueeze(-2)
            - positions.unsqueeze(-3),
            dim=-1,
        )
        return dists <= self.comm_range

    def _distribution(self, raw_mu):
        std = float(getattr(self.cfg, "ppo_std", 0.7))

        if std <= 0:
            raise ValueError("ppo_std は正の値にしてください。")

        base = Independent(
            Normal(
                raw_mu,
                torch.full_like(raw_mu, std),
            ),
            1,  # action_dim を１エージェントの行動として扱う
        )

        return TransformedDistribution(
            base,
            [SigmoidTransform(cache_size=1)],
        )

    @torch.no_grad()
    def plan(
        self,
        obs,
        positions=None,
        eval_mode=False,
        step=None,
        t0=True,
    ):
        obs_t = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        pos_t = (
            np.expand_dims(positions, axis=0)
            if positions is not None
            else None
        )
        adj_mask = self._make_adj_mask(pos_t)

        raw_mu = self.actor(obs_t, adj_mask)

        if eval_mode:
            # 分布の変換後の中心を決定論的行動として使用。
            return torch.sigmoid(raw_mu).squeeze(0)

        dist = self._distribution(raw_mu)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(obs_t)

        # 行動を clamp しない。
        # 保存する log_prob は、環境に渡す action のもの。
        return (
            action.squeeze(0),
            log_prob.squeeze(0),
            value.squeeze(0),
        )

    def update(
        self,
        next_obs_tensor,
        next_pos_tensor,
        next_done,
    ):
        if not self.buffer.rewards:
            raise ValueError("MAPPO rollout buffer が空です。")

        # next_pos_tensor と next_done は、既存の
        # train.py との呼び出し互換性のため受け取る。
        # 終了判定には各遷移で保存した masks[t] を使う。
        _ = next_pos_tensor, next_done

        obs = torch.stack(self.buffer.obs)              # [T, N, obs_dim]
        actions = torch.stack(self.buffer.actions)      # [T, N, action_dim]
        old_log_probs = torch.stack(
            self.buffer.log_probs
        )                                                 # [T, N]
        rewards = torch.stack(self.buffer.rewards)      # [T, N]
        values = torch.stack(self.buffer.values)        # [T, N]
        masks = torch.stack(self.buffer.masks)          # [T]

        T = rewards.shape[0]

        if rewards.shape != (T, self.N):
            raise ValueError(
                "MAPPO の rewards は [T, N] を想定します。"
                f"実際の shape: {tuple(rewards.shape)}"
            )

        # 位置情報は通信を使う Actor の更新に必要。
        if self.cfg.comm_type == "local":
            if any(p is None for p in self.buffer.positions):
                raise ValueError(
                    "局所通信の rollout に位置情報がありません。"
                )
            positions = torch.stack(
                self.buffer.positions
            )                                             # [T, N, 2]
        else:
            positions = None

        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(
                next_obs_tensor,
                dtype=torch.float32,
                device=self.device,
            )

            next_value = self.critic(
                next_obs_tensor.unsqueeze(0)
            ).squeeze(0)                                  # [N]

            advantages = torch.zeros_like(rewards)
            lastgaelam = torch.zeros_like(rewards[0])

            for t in reversed(range(T)):
                nextvalues = (
                    next_value
                    if t == T - 1
                    else values[t + 1]
                )

                # masks[t] は「時刻 t の遷移後に継続するか」。
                # masks[t + 1] ではない。
                nextnonterminal = masks[t]

                delta = (
                    rewards[t]
                    + self.cfg.discount
                    * nextvalues
                    * nextnonterminal
                    - values[t]
                )

                lastgaelam = (
                    delta
                    + self.cfg.discount
                    * self.cfg.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
                advantages[t] = lastgaelam

            returns = advantages + values

            # 全時刻・全エージェントで正規化。
            advantages = (
                advantages - advantages.mean()
            ) / (
                advantages.std(unbiased=False) + 1e-8
            )

        minibatch_size = int(
            getattr(
                self.cfg,
                "ppo_minibatch_size",
                64,
            )
        )
        minibatch_size = min(minibatch_size, T)

        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        num_minibatches = 0

        for _ in range(self.cfg.ppo_epochs):
            # ミニバッチは「時刻」を単位に切る。
            # 同時刻の全エージェントは一緒に扱う。
            permutation = torch.randperm(
                T,
                device=self.device,
            )

            for start in range(0, T, minibatch_size):
                idx = permutation[
                    start:start + minibatch_size
                ]

                b_obs = obs[idx]
                b_actions = actions[idx]
                b_old_log_probs = old_log_probs[idx]
                b_advantages = advantages[idx]
                b_returns = returns[idx]

                adj_mask = (
                    self._make_adj_mask(positions[idx])
                    if positions is not None
                    else None
                )

                raw_mu = self.actor(b_obs, adj_mask)
                dist = self._distribution(raw_mu)

                new_log_probs = dist.log_prob(b_actions)

                # TransformedDistribution.entropy() は
                # 一般には実装されていないため、
                # 再パラメータ化サンプルによる推定を使用。
                entropy_estimate = (
                    -dist.log_prob(dist.rsample()).mean()
                )

                ratio = torch.exp(
                    new_log_probs - b_old_log_probs
                )

                surr1 = ratio * b_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.cfg.ppo_clip_param,
                        1.0 + self.cfg.ppo_clip_param,
                    )
                    * b_advantages
                )

                actor_loss = (
                    -torch.min(surr1, surr2).mean()
                    - self.cfg.entropy_coef
                    * entropy_estimate
                )

                new_values = self.critic(b_obs)
                critic_loss = F.mse_loss(
                    new_values,
                    b_returns,
                )

                self.actor_optim.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.cfg.grad_clip_norm,
                )
                self.actor_optim.step()

                self.critic_optim.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.cfg.grad_clip_norm,
                )
                self.critic_optim.step()

                actor_loss_sum += actor_loss.item()
                critic_loss_sum += critic_loss.item()
                num_minibatches += 1

        self.buffer.clear()

        return {
            "actor_loss": (
                actor_loss_sum / num_minibatches
            ),
            "critic_loss": (
                critic_loss_sum / num_minibatches
            ),
        }

    def save(self, filepath):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            filepath,
        )

    def load(self, filepath):
        checkpoint = torch.load(
            filepath,
            map_location=self.device,
        )
        self.actor.load_state_dict(
            checkpoint["actor"]
        )
        self.critic.load_state_dict(
            checkpoint["critic"]
        )