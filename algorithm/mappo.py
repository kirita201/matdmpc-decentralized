import torch
import torch.nn.functional as F
from torch.distributions import Beta

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
        self.positions = []

    def store(self, obs, action, log_prob, reward, value, pos):
        self.obs.append(obs.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward.detach())
        self.values.append(value.detach())
        self.positions.append(pos.detach() if pos is not None else None)


class MAPPO:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.comm_range = (
            float(getattr(cfg, "comm_range", "inf"))
            if cfg.comm_type == "local"
            else float("inf")
        )

        self.actor = MFActor(cfg).to(self.device)
        self.critic = MFCritic(cfg, use_action=False).to(self.device)

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.lr
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.lr
        )

        self.buffer = RolloutBuffer()
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

    def _action_dist(self, obs, adj_mask):
        # MFActor の出力は (0, 1) のテンソル。
        mu = self.actor(obs, adj_mask)

        # 正のパラメータを作る。ppo_concentration は分布の
        # 集中度を調節するハイパーパラメータ。
        concentration = float(
            getattr(self.cfg, "ppo_concentration", 20.0)
        )
        if concentration <= 0:
            raise ValueError("ppo_concentration must be positive.")

        alpha = 1.0 + concentration * mu
        beta = 1.0 + concentration * (1.0 - mu)
        return Beta(alpha, beta)

    @torch.no_grad()
    def plan(
        self, obs, positions=None, eval_mode=False, step=None, t0=True
    ):
        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        pos_t = (
            torch.as_tensor(
                positions, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            if positions is not None
            else None
        )
        adj_mask = self._make_adj_mask(pos_t)
        dist = self._action_dist(obs_t, adj_mask)

        if eval_mode:
            return dist.mean.squeeze(0)

        # Beta.sample() が返す範囲内の行動を、そのまま環境に渡す。
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs_t)

        return (
            action.squeeze(0),
            log_prob.squeeze(0),
            value.squeeze(0),
        )

    def update(self, next_obs_tensor):
        if not self.buffer.rewards:
            raise ValueError("MAPPO rollout buffer is empty.")

        # この実装ではバッファに１エピソード分だけ入っている。
        obs = torch.stack(self.buffer.obs)                # [T, N, obs_dim]
        actions = torch.stack(self.buffer.actions)        # [T, N, act_dim]
        old_log_probs = torch.stack(self.buffer.log_probs) # [T, N]
        rewards = torch.stack(self.buffer.rewards)        # [T, N]
        values = torch.stack(self.buffer.values)          # [T, N]

        positions = (
            torch.stack(self.buffer.positions)
            if self.buffer.positions[0] is not None
            else None
        )

        with torch.no_grad():
            # 時間切れは真の終端ではない。最後の観測でブートストラップ。
            next_value = self.critic(
                next_obs_tensor.unsqueeze(0)
            ).squeeze(0)

            advantages = torch.empty_like(rewards)
            last_advantage = torch.zeros_like(next_value)

            for t in reversed(range(len(rewards))):
                value_after = (
                    next_value
                    if t == len(rewards) - 1
                    else values[t + 1]
                )

                delta = (
                    rewards[t]
                    + self.cfg.discount * value_after
                    - values[t]
                )
                last_advantage = (
                    delta
                    + self.cfg.discount
                    * self.cfg.gae_lambda
                    * last_advantage
                )
                advantages[t] = last_advantage

            returns = advantages + values

            # PPOの最適化を安定させるため、収集済みデータ上で正規化。
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std(unbiased=False) + 1e-8)

        rollout_size = len(rewards)
        minibatch_size = int(
            getattr(self.cfg, "ppo_minibatch_size", rollout_size)
        )
        if minibatch_size <= 0:
            raise ValueError("ppo_minibatch_size must be positive.")

        total_a_loss = 0.0
        total_c_loss = 0.0
        num_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            permutation = torch.randperm(
                rollout_size, device=self.device
            )

            for start in range(0, rollout_size, minibatch_size):
                batch_idx = permutation[start:start + minibatch_size]

                batch_pos = (
                    positions[batch_idx]
                    if positions is not None
                    else None
                )
                adj_mask = self._make_adj_mask(batch_pos)
                dist = self._action_dist(obs[batch_idx], adj_mask)

                new_log_probs = dist.log_prob(
                    actions[batch_idx]
                ).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(
                    new_log_probs - old_log_probs[batch_idx]
                )
                batch_advantages = advantages[batch_idx]

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.cfg.ppo_clip_param,
                    1.0 + self.cfg.ppo_clip_param,
                ) * batch_advantages

                actor_loss = (
                    -torch.min(surr1, surr2).mean()
                    - self.cfg.entropy_coef * entropy
                )

                new_values = self.critic(obs[batch_idx])
                critic_loss = F.mse_loss(
                    new_values, returns[batch_idx]
                )

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.cfg.grad_clip_norm
                )
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.cfg.grad_clip_norm
                )
                self.critic_optim.step()

                total_a_loss += actor_loss.item()
                total_c_loss += critic_loss.item()
                num_updates += 1

        self.buffer.clear()

        return {
            "actor_loss": total_a_loss / num_updates,
            "critic_loss": total_c_loss / num_updates,
        }

    def save(self, filepath):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
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

        if "actor_optim" in checkpoint:
            self.actor_optim.load_state_dict(checkpoint["actor_optim"])
        if "critic_optim" in checkpoint:
            self.critic_optim.load_state_dict(checkpoint["critic_optim"])