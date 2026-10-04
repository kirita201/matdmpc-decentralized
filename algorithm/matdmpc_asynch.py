# algorithm/tdmpc_asynch.py
import numpy as np
import torch
from copy import deepcopy
import algorithm.helper as h
from algorithm.models import MACLM  # モデルファイルを models.py に統一することを想定
import torch.nn.functional as F

class AsynchMATDMPC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.comm_range = float(getattr(cfg, 'comm_range', 'inf'))

        model = MACLM(cfg).to(self.device)
        if hasattr(torch, 'compile'):
            self.model = torch.compile(model)
        else:
            self.model = model
        self.model_target = deepcopy(self.model)
        
        model_params = [p for name, p in self.model.named_parameters() if '_pi' not in name]
        self.optim = torch.optim.Adam(model_params, lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
        
        self.model.eval()
        self.model_target.eval()
        self.N = cfg.num_agents

        # 暫定行動系列のトラッキング [N, horizon, action_dim]
        self._prev_mean = torch.zeros(self.N, self.cfg.horizon, self.cfg.action_dim, device=self.device)

    def _make_adj_mask(self, positions):
        """
        positions: Tensor [..., N, 2] (NumPyまたはTensorから変換された絶対座標)
        Returns: BoolTensor [..., N, N] または 全て通信可能な場合は None
        """
        if self.comm_range == float('inf') or positions is None:
            return None
        
        # Tensor化とデバイス転送の保証
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
        else:
            positions = positions.to(self.device, dtype=torch.float32)

        # ペアワイズ距離の計算
        # positions: [..., N, 1, 2] vs [..., 1, N, 2] -> dists: [..., N, N]
        dists = torch.norm(positions.unsqueeze(-2) - positions.unsqueeze(-3), dim=-1)
        return dists <= self.comm_range

    @torch.no_grad()
    def estimate_value(self, e0, actions, horizon, adj_mask=None):
        """マルチエージェントの将来軌道を評価 (adj_mask=None の時は全体通信)"""
        G, discount = torch.zeros(actions.size(1), self.N, device=self.device), 1.0
        e = e0
        for t in range(horizon):
            z = self.model.communicate(e, actions[t], adj_mask=adj_mask)
            e, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
            
        # 終端Q値の計算
        last_action = actions[-1]
        z_pre = self.model.communicate(e, last_action, adj_mask=adj_mask)
        pi_a = self.model.pi(z_pre, self.cfg.min_std)
        z_eval = self.model.communicate(e, pi_a, adj_mask=adj_mask)
        
        q1, q2 = self.model.Q(z_eval, pi_a)
        G += discount * torch.min(q1, q2)
        return G

    @torch.no_grad()
    def plan(self, obs, positions=None, eval_mode=False, step=None, t0=True):
        """エージェントごとの非同期プランニング (他アクターの行動は前回解で仮置き)"""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, N, obs_dim]
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.N, self.cfg.action_dim, device=self.device).uniform_(0, 1)

        e0 = self.model.encode(obs_t)  # [1, N, latent]

        # ── 絶対座標から初期通信グラフを生成（ホライズン内固定） ──
        adj_mask_1 = self._make_adj_mask(np.expand_dims(positions, axis=0) if positions is not None else None) # [1, N, N] または None

        # --- prev_mean (前回導出解) の更新 ---
        if t0:
            self._prev_mean.zero_()
        else:
            self._prev_mean[:, :-1] = self._prev_mean[:, 1:].clone()

            # 保存している系列の末尾を、一個前の行動で埋める
            if self.cfg.horizon >= 2:
                self._prev_mean[:, -1] = self._prev_mean[:, -2]

            # 今回使う horizon が cfg.horizon より短い場合も、
            # 実際に使う暫定系列の末尾を一個前と同じにする
            if horizon >= 2:
                self._prev_mean[:, horizon - 1] = self._prev_mean[:, horizon - 2]

        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        total_samples = self.cfg.num_samples + num_pi_trajs
        N, H, A = self.N, horizon, self.cfg.action_dim
        B = total_samples

        mean = self._prev_mean[:, :H].clone()
        std  = 2 * torch.ones(N, H, A, device=self.device)
        e_batch = e0.repeat(B, 1, 1)

        adj_mask_B = adj_mask_1.expand(B, N, N) if adj_mask_1 is not None else None

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for i in range(self.cfg.iterations):
                # 1. ランダムサンプル
                noise = torch.randn(H, N, self.cfg.num_samples, A, device=self.device)
                a_random = torch.clamp(mean.permute(1, 0, 2).unsqueeze(2) + std.permute(1, 0, 2).unsqueeze(2) * noise, 0, 1)

                # 2. Policy サンプル
                if num_pi_trajs > 0:
                    S = num_pi_trajs
                    a_pi = torch.empty(H, N, S, A, device=self.device)

                    # e_views[i, s]: エージェントi視点で予測した候補sの状態
                    e_views = (
                        e0.unsqueeze(0)
                        .expand(N, S, N, -1)
                        .clone()
                    )  # [N, S, N, latent]

                    if adj_mask_1 is not None:
                        view_masks_pi = (
                            self._make_view_masks(adj_mask_1)
                            .expand(N, S, N, N)
                        )
                    else:
                        view_masks_pi = None

                    idx = torch.arange(N, device=self.device)

                    for t in range(H):
                        # 方策入力の仮行動には従来どおりprev_meanを使用
                        a_guess = (
                            self._prev_mean[:, t]
                            .unsqueeze(0)
                            .expand(S, N, A)
                        )  # [S, N, A]

                        z_views = self.model.communicate_per_agent(
                            e_views,
                            a_guess,
                            adj_mask_per_agent=view_masks_pi,
                        )  # [N, S, N, latent]

                        pi_views = self.model.pi(z_views, self.std)
                        a_self = pi_views[idx, :, idx, :]  # [N, S, A]
                        a_pi[t] = a_self

                        # 各エージェントが自分の視点で出した行動を組み合わせる
                        a_joint = a_self.permute(1, 0, 2)  # [S, N, A]

                        z_next = self.model.communicate_per_agent(
                            e_views,
                            a_joint,
                            adj_mask_per_agent=view_masks_pi,
                        )

                        z_flat = z_next.reshape(N * S, N, -1)
                        a_flat = (
                            a_joint.unsqueeze(0)
                            .expand(N, S, N, A)
                            .reshape(N * S, N, A)
                        )

                        next_e_flat, _ = self.model.next(z_flat, a_flat)
                        e_views = next_e_flat.reshape(N, S, N, -1)

                    actions_all = torch.cat([a_random, a_pi], dim=2)
                else:
                    actions_all = a_random

                # 3. 自己中心的な Joint actions の構築
                joint_base = self._prev_mean[:, :H].permute(1, 0, 2).unsqueeze(1).expand(H, B, N, A)
                joint_all = joint_base.unsqueeze(0).expand(N, H, B, N, A).clone()
                agent_idx_t = torch.arange(N, device=self.device)
                joint_all[agent_idx_t, :, :, agent_idx_t, :] = actions_all.permute(1, 0, 2, 3)

                # 4. 一括評価用展開
                e_all = e0.repeat(N * B, 1, 1)
                joint_flat = joint_all.permute(1, 0, 2, 3, 4).reshape(H, N * B, N, A)
                if adj_mask_1 is not None:
                    # [N, 1, N, N] -> [N, B, N, N] -> [N * B, N, N]
                    view_masks_1 = self._make_view_masks(adj_mask_1)
                    adj_mask_NB = (
                        view_masks_1.expand(N, B, N, N)
                        .reshape(N * B, N, N)
                    )
                else:
                    adj_mask_NB = None

                values_flat = self.estimate_value(e_all, joint_flat, horizon, adj_mask=adj_mask_NB)
                values = torch.stack([values_flat[n * B:(n + 1) * B, n] for n in range(N)], dim=0)

                # 5. CEM 更新
                elite_idxs = torch.topk(values, self.cfg.num_elites, dim=1).indices
                elite_actions = torch.stack([actions_all[:, n, elite_idxs[n], :] for n in range(N)], dim=0)
                elite_value = torch.stack([values[n, elite_idxs[n]] for n in range(N)], dim=0).float()
                
                max_value = elite_value.max(dim=1, keepdim=True).values
                score = torch.exp(self.cfg.temperature * (elite_value - max_value))
                score = score / (score.sum(dim=1, keepdim=True) + 1e-8)

                w = score.unsqueeze(1).unsqueeze(-1)
                _mean = (w * elite_actions).sum(dim=2)
                _std  = torch.sqrt((w * (elite_actions - _mean.unsqueeze(2)) ** 2).sum(dim=2)).clamp_(self.std, 2)

                mean = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean
                std  = _std

        self._prev_mean[:, :H] = mean
        score_np = score.float().cpu().numpy()
        final_actions = torch.zeros(N, A, device=self.device)
        for n in range(N):
            best_idx = np.random.choice(self.cfg.num_elites, p=score_np[n])
            a = elite_actions[n, 0, best_idx]
            if not eval_mode:
                a = a + std[n, 0] * torch.randn(A, device=self.device)
            final_actions[n] = a.clamp(0, 1)

        return final_actions

    def update(self, replay_buffer, step):
        """価値・方策は実観測 latent、モデル損失は予測 latent で学習する。"""
        beta = h.linear_schedule(self.cfg.per_beta, step)
        obs, next_obses, action, reward, positions, idxs, weights = (
            replay_buffer.sample(beta)
        )

        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        N = self.N
        B = obs.shape[0]
        H = self.cfg.horizon
        agent_idx = torch.arange(N, device=self.device)

        # 実観測の価値計算には、対応する時刻の通信グラフを使用する。
        # value_adj[t]:      時刻 t の Q / actor 用
        # value_adj[t + 1]:  時刻 t + 1 の TD target 用
        with torch.no_grad():
            value_adj = [
                self._make_adj_mask(positions[t])
                for t in range(H + 1)
            ]

            # 世界モデルの予測経路は従来どおり初期グラフで固定する。
            rollout_adj = value_adj[0]

            value_view_adj = [
                self._make_view_masks(mask) if mask is not None else None
                for mask in value_adj
            ]
            rollout_view_adj = value_view_adj[0]

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        ):
            e0 = self.model.encode(obs)

            # actor 更新では、critic 更新前に計算した実観測 latent を
            # detach して使用する。これまでの es_* と同じ扱い。
            actor_states = []

            consistency_loss = 0
            reward_loss = 0
            value_loss = 0
            priority_loss = 0

            # ============================================================
            # 全体通信
            # ============================================================
            if rollout_adj is None:
                e_roll = e0

                for t in range(H):
                    # ---------- 実観測経路：critic ----------
                    current_obs = obs if t == 0 else next_obses[t - 1]
                    e_real = e0 if t == 0 else self.model.encode(current_obs)
                    actor_states.append(e_real.detach())

                    z_real = self.model.communicate(
                        e_real,
                        action[t],
                        adj_mask=None,
                    )
                    q1, q2 = self.model.Q(z_real, action[t])
                    q_joint1, q_joint2 = self.model.Q_joint(
                        q1, q2, e_real
                    )

                    # ---------- 予測経路：世界モデル ----------
                    z_roll = self.model.communicate(
                        e_roll,
                        action[t],
                        adj_mask=None,
                    )
                    e_roll, reward_pred = self.model.next(
                        z_roll, action[t]
                    )

                    # ---------- 実観測経路：TD target ----------
                    with torch.no_grad():
                        next_e_target = self.model_target.encode(
                            next_obses[t]
                        )

                        next_z_for_pi = self.model.communicate(
                            next_e_target,
                            action[t + 1],
                            adj_mask=None,
                        )
                        next_a = self.model.pi(
                            next_z_for_pi,
                            self.cfg.min_std,
                        )

                        next_z_for_q = self.model_target.communicate(
                            next_e_target,
                            next_a,
                            adj_mask=None,
                        )
                        nq1, nq2 = self.model_target.Q(
                            next_z_for_q, next_a
                        )
                        nq_joint1, nq_joint2 = self.model_target.Q_joint(
                            nq1, nq2, next_e_target
                        )
                        nq_joint = torch.min(nq_joint1, nq_joint2)

                        joint_reward = (
                            reward[t][:, :1]
                            if getattr(
                                self.cfg, "reward_type", "individual"
                            ) == "global"
                            else reward[t].sum(dim=1, keepdim=True)
                        )
                        td_target = (
                            joint_reward
                            + self.cfg.discount * nq_joint
                        )

                    rho = self.cfg.rho ** t

                    consistency_loss += rho * h.mse(
                        e_roll,
                        next_e_target,
                        reduce=False,
                    ).mean(dim=(1, 2))

                    reward_loss += rho * h.mse(
                        reward_pred,
                        reward[t],
                        reduce=False,
                    ).mean(dim=1)

                    value_loss += rho * (
                        h.mse(
                            q_joint1.float(),
                            td_target.float(),
                            reduce=False,
                        )
                        + h.mse(
                            q_joint2.float(),
                            td_target.float(),
                            reduce=False,
                        )
                    ).squeeze(-1)

                    priority_loss += rho * (
                        h.l1(q_joint1, td_target, reduce=False)
                        + h.l1(q_joint2, td_target, reduce=False)
                    ).squeeze(-1)

            # ============================================================
            # 局所通信
            # ============================================================
            else:
                # 予測経路。各エージェント視点の rollout を維持する。
                e_roll_views = (
                    e0.unsqueeze(0)
                    .expand(N, -1, -1, -1)
                    .contiguous()
                )

                # 世界モデルの損失のマスクも従来どおり初期グラフ。
                neighbor_mask = rollout_adj.permute(
                    1, 0, 2
                ).float()  # [N, B, N]
                neighbor_count = (
                    neighbor_mask.sum(dim=2)
                    .clamp(min=1.0)
                )  # [N, B]

                for t in range(H):
                    # ---------- 実観測経路：critic ----------
                    current_obs = obs if t == 0 else next_obses[t - 1]
                    e_real = e0 if t == 0 else self.model.encode(current_obs)

                    e_real_views = (
                        e_real.unsqueeze(0)
                        .expand(N, -1, -1, -1)
                        .contiguous()
                    )  # [N, B, N, latent]

                    # actor も同じ時刻の実観測・通信グラフを使用する。
                    actor_states.append(e_real_views.detach())

                    z_real_views = self.model.communicate_per_agent(
                        e_real_views,
                        action[t],
                        adj_mask_per_agent=value_view_adj[t],
                    )

                    z_real_self = z_real_views[
                        agent_idx, :, agent_idx, :
                    ].permute(1, 0, 2)  # [B, N, latent]

                    q1, q2 = self.model.Q(
                        z_real_self,
                        action[t],
                    )
                    q_joint1, q_joint2 = self.model.Q_joint(
                        q1, q2, e_real
                    )

                    # ---------- 予測経路：世界モデル ----------
                    z_roll_views = self.model.communicate_per_agent(
                        e_roll_views,
                        action[t],
                        adj_mask_per_agent=rollout_view_adj,
                    )

                    z_roll_flat = z_roll_views.reshape(
                        N * B, N, -1
                    )
                    a_roll_flat = (
                        action[t]
                        .unsqueeze(0)
                        .expand(N, -1, -1, -1)
                        .reshape(N * B, N, -1)
                    )

                    next_e_flat, reward_pred_flat = self.model.next(
                        z_roll_flat,
                        a_roll_flat,
                    )
                    next_e_roll_views = next_e_flat.reshape(
                        N, B, N, -1
                    )
                    reward_pred_views = reward_pred_flat.reshape(
                        N, B, N
                    )

                    # ---------- 実観測経路：TD target ----------
                    with torch.no_grad():
                        next_e_target = self.model_target.encode(
                            next_obses[t]
                        )
                        next_e_target_views = (
                            next_e_target.unsqueeze(0)
                            .expand(N, -1, -1, -1)
                            .contiguous()
                        )

                        next_z_for_pi_views = (
                            self.model_target.communicate_per_agent(
                                next_e_target_views,
                                action[t + 1],
                                adj_mask_per_agent=value_view_adj[t + 1],
                            )
                        )
                        next_z_for_pi_self = next_z_for_pi_views[
                            agent_idx, :, agent_idx, :
                        ].permute(1, 0, 2)

                        next_a = self.model.pi(
                            next_z_for_pi_self,
                            self.cfg.min_std,
                        )

                        next_z_for_q_views = (
                            self.model_target.communicate_per_agent(
                                next_e_target_views,
                                next_a,
                                adj_mask_per_agent=value_view_adj[t + 1],
                            )
                        )
                        next_z_for_q_self = next_z_for_q_views[
                            agent_idx, :, agent_idx, :
                        ].permute(1, 0, 2)

                        nq1, nq2 = self.model_target.Q(
                            next_z_for_q_self, next_a
                        )
                        nq_joint1, nq_joint2 = self.model_target.Q_joint(
                            nq1, nq2, next_e_target
                        )
                        nq_joint = torch.min(nq_joint1, nq_joint2)

                        joint_reward = (
                            reward[t][:, :1]
                            if getattr(
                                self.cfg, "reward_type", "individual"
                            ) == "global"
                            else reward[t].sum(dim=1, keepdim=True)
                        )
                        td_target = (
                            joint_reward
                            + self.cfg.discount * nq_joint
                        )

                    # 次の世界モデル予測へ進む。
                    e_roll_views = next_e_roll_views
                    rho = self.cfg.rho ** t

                    # ---------- 世界モデルの損失 ----------
                    # consistency_raw: [N, B, N]
                    # 軸の意味:
                    #   0: 視点となるエージェント i
                    #   1: replay buffer のサンプル b
                    #   2: 予測対象のエージェント j
                    consistency_raw = h.mse(
                        next_e_roll_views,
                        next_e_target.unsqueeze(0).expand(
                            N, -1, -1, -1
                        ),
                        reduce=False,
                    ).mean(dim=-1)  # [N, B, N]

                    # 対角マスク [N, 1, N]
                    # self_mask[i, 0, j] = True iff i == j
                    self_mask = torch.eye(
                        N,
                        dtype=neighbor_mask.dtype,
                        device=neighbor_mask.device,
                    ).unsqueeze(1)

                    # ----- Consistency: 自分自身の予測誤差 -----
                    # 各視点 i の「自分自身 i」の誤差を取得する。
                    # 自分自身の損失は近傍数で割らず、独立して維持する。
                    consistency_self = (
                        consistency_raw * self_mask
                    ).sum(dim=2)  # [N, B]

                    consistency_self = consistency_self.mean(dim=0)  # [B]

                    # ----- Consistency: 近傍他者の予測誤差 -----
                    # 自分自身を除き、初期通信グラフの近傍だけを対象にする。
                    other_neighbor_mask = neighbor_mask * (1.0 - self_mask)

                    other_neighbor_count = (
                        other_neighbor_mask.sum(dim=2)
                        .clamp(min=1.0)
                    )  # [N, B]

                    consistency_other = (
                        (consistency_raw * other_neighbor_mask).sum(dim=2)
                        / other_neighbor_count
                    )  # [N, B]

                    # 自分以外に近傍がいない場合は、
                    # マスクの積がゼロなので損失もゼロになる。
                    consistency_other = consistency_other.mean(dim=0)  # [B]

                    neighbor_consistency_coef = float(
                        getattr(
                            self.cfg,
                            "neighbor_consistency_coef",
                            0.5,
                        )
                    )

                    consistency_loss += rho * (
                        consistency_self
                        + neighbor_consistency_coef * consistency_other
                    )

                    # ----- Reward prediction loss -----
                    reward_target_views = (
                        reward[t].unsqueeze(0).expand(N, -1, -1)
                    )  # [N, B, N]

                    reward_raw = h.mse(
                        reward_pred_views,
                        reward_target_views,
                        reduce=False,
                    )  # [N, B, N]

                    reward_type = getattr(
                        self.cfg,
                        "reward_type",
                        "individual",
                    )

                    if reward_type == "individual":
                        # 個別報酬の場合は、各視点 i から見た
                        # エージェント i 自身の報酬予測だけを学習する。
                        reward_self = (
                            reward_raw * self_mask
                        ).sum(dim=2)  # [N, B]

                        # 各サンプルについて、エージェント間で平均する。
                        reward_loss += rho * reward_self.mean(dim=0)  # [B]

                    else:
                        # 全体報酬の場合は従来どおり、
                        # 各視点の近傍内にある報酬予測誤差を平均する。
                        reward_masked = (
                            (reward_raw * neighbor_mask).sum(dim=2)
                            / neighbor_count
                        )  # [N, B]

                        reward_loss += rho * reward_masked.mean(dim=0)  # [B]

                    # ---------- critic の損失 ----------
                    value_loss += rho * (
                        h.mse(
                            q_joint1.float(),
                            td_target.float(),
                            reduce=False,
                        )
                        + h.mse(
                            q_joint2.float(),
                            td_target.float(),
                            reduce=False,
                        )
                    ).squeeze(-1)

                    priority_loss += rho * (
                        h.l1(q_joint1, td_target, reduce=False)
                        + h.l1(q_joint2, td_target, reduce=False)
                    ).squeeze(-1)

            total_loss = (
                self.cfg.consistency_coef * consistency_loss
                + self.cfg.reward_coef * reward_loss
                + self.cfg.value_coef * value_loss
            )

            weighted_loss = (total_loss * weights).mean()
            weighted_loss.register_hook(
                lambda grad: grad * (1 / H)
            )

        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.grad_clip_norm,
        )
        self.optim.step()

        p_loss = priority_loss.clamp(max=1e4).float().detach()
        safe_p_loss = torch.nan_to_num(p_loss, nan=1.0)
        replay_buffer.update_priorities(idxs, safe_p_loss)

        # ================================================================
        # Actor 更新：実観測 latent と、その時刻の通信グラフを使用
        # ================================================================
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        try:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
            ):
                pi_loss = 0

                for t in range(H):
                    if rollout_adj is None:
                        e_real = actor_states[t]

                        z_pre = self.model.communicate(
                            e_real,
                            action[t],
                            adj_mask=None,
                        )
                        a_t = self.model.pi(z_pre, 0)

                        z_eval = self.model.communicate(
                            e_real,
                            a_t,
                            adj_mask=None,
                        )
                        q1, q2 = self.model.Q(z_eval, a_t)

                    else:
                        e_real_views = actor_states[t]
                        mask_t = value_view_adj[t]

                        z_pre_views = (
                            self.model.communicate_per_agent(
                                e_real_views,
                                action[t],
                                adj_mask_per_agent=mask_t,
                            )
                        )
                        z_pre_self = z_pre_views[
                            agent_idx, :, agent_idx, :
                        ]  # [N, B, latent]

                        a_self = self.model.pi(z_pre_self, 0)
                        a_t = a_self.permute(1, 0, 2)

                        z_eval_views = (
                            self.model.communicate_per_agent(
                                e_real_views,
                                a_t,
                                adj_mask_per_agent=mask_t,
                            )
                        )
                        z_eval_self = z_eval_views[
                            agent_idx, :, agent_idx, :
                        ].permute(1, 0, 2)

                        q1, q2 = self.model.Q(
                            z_eval_self, a_t
                        )

                    pi_loss += (
                        -torch.min(q1, q2).mean()
                        * (self.cfg.rho ** t)
                    )

            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model._pi.parameters(),
                self.cfg.grad_clip_norm,
            )
            self.pi_optim.step()

        finally:
            self.model.track_q_grad(True)

        if step % self.cfg.update_freq == 0:
            h.ema(
                self.model,
                self.model_target,
                self.cfg.tau,
            )

        self.model.eval()
        return {
            "total_loss": total_loss.mean().detach(),
            "consistency_loss": consistency_loss.mean().detach(),
            "reward_loss": reward_loss.mean().detach(),
            "value_loss": value_loss.mean().detach(),
            "pi_loss": pi_loss.detach(),
        }

    def save(self, filepath):
        state = {
            'model':        self.model.state_dict(),
            'model_target': self.model_target.state_dict(),
            'optim':        self.optim.state_dict(),
            'pi_optim':     self.pi_optim.state_dict(),
        }
        torch.save(state, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model_target.load_state_dict(checkpoint['model_target'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.pi_optim.load_state_dict(checkpoint['pi_optim'])
        print(f"[AsynchMATDMPC] Loaded checkpoints from {filepath}")

    def _make_view_masks(self, adj_mask_global):
        """
        adj_mask_global: [B, N, N]
            [b, j, k] = jがkを直接参照できるか

        戻り値: [N, B, N, N]
            [i, b, j, k] = i視点の予測において、
                        jがkを参照できるか

        i視点では、iの近傍集合内にあるj, kについてのみ
        元の距離グラフの辺を維持する。
        集合外のトークンは自分自身だけ参照可能にする。
        """
        if adj_mask_global is None:
            return None

        N = adj_mask_global.size(-1)

        # members[i, b, j]:
        # サンプルbにおいて、jが視点iの近傍に含まれるか
        members = adj_mask_global.permute(1, 0, 2)  # [N, B, N]

        # 視点iの近傍集合内のj, kに限定し、
        # その中でも元のグラフで許された通信だけを残す
        view_masks = (
            adj_mask_global.unsqueeze(0)    # [1, B, N, N]
            & members.unsqueeze(-1)         # [N, B, N, 1]: jが近傍
            & members.unsqueeze(-2)         # [N, B, 1, N]: kが近傍
        )

        # 近傍外のトークンも含め、各トークンの自己参照は許可。
        # 参照先ゼロの行を防ぎ、NaNを避ける。
        eye = torch.eye(N, dtype=torch.bool, device=adj_mask_global.device)
        view_masks = view_masks | eye.view(1, 1, N, N)

        return view_masks