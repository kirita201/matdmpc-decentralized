from copy import deepcopy

import numpy as np
import torch

import algorithm.helper as h
from algorithm.models import MACLM


class AsynchMATDMPC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.comm_range = float(getattr(cfg, "comm_range", "inf"))
        self.diagnostics = None

        self.joint_value_coef = float(
            getattr(
                cfg,
                "joint_value_coef",
                getattr(cfg, "value_coef", 1.0),
            )
        )
        self.individual_value_coef = float(
            getattr(cfg, "individual_value_coef", 0.0)
        )

        model = MACLM(cfg).to(self.device)
        if hasattr(torch, "compile"):
            self.model = torch.compile(model)
        else:
            self.model = model
        self.model_target = deepcopy(self.model)

        model_params = [
            p
            for name, p in self.model.named_parameters()
            if "_pi" not in name
        ]
        self.optim = torch.optim.Adam(
            model_params,
            lr=self.cfg.lr,
        )
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(),
            lr=self.cfg.lr,
        )

        self.model.eval()
        self.model_target.eval()
        self.N = cfg.num_agents

        self._prev_mean = torch.zeros(
            self.N,
            self.cfg.horizon,
            self.cfg.action_dim,
            device=self.device,
        )

    def _make_adj_mask(self, positions):
        """
        positions: [..., N, 2]
        戻り値: [..., N, N] の bool テンソル。
        全体通信なら None。
        """
        if self.comm_range == float("inf") or positions is None:
            return None

        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(
                positions,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            positions = positions.to(
                self.device,
                dtype=torch.float32,
            )

        dists = torch.norm(
            positions.unsqueeze(-2) - positions.unsqueeze(-3),
            dim=-1,
        )
        return dists <= self.comm_range

    @torch.no_grad()
    def estimate_value(self, e0, actions, horizon, adj_mask=None):
        G = torch.zeros(
            actions.size(1),
            self.N,
            device=self.device,
        )
        discount = 1.0
        e = e0

        for t in range(horizon):
            z = self.model.communicate(
                e,
                actions[t],
                adj_mask=adj_mask,
            )
            e, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount

        last_action = actions[-1]
        z_pre = self.model.communicate(
            e,
            last_action,
            adj_mask=adj_mask,
        )
        pi_a = self.model.pi(z_pre, self.cfg.min_std)
        z_eval = self.model.communicate(
            e,
            pi_a,
            adj_mask=adj_mask,
        )

        q1, q2 = self.model.Q(z_eval, pi_a)
        G += discount * torch.min(q1, q2)
        return G

    @torch.no_grad()
    def plan(self, obs, positions=None, eval_mode=False, step=None, t0=True):
        obs_t = torch.tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        horizon = int(
            min(
                self.cfg.horizon,
                h.linear_schedule(self.cfg.horizon_schedule, step),
            )
        )

        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(
                self.N,
                self.cfg.action_dim,
                device=self.device,
            ).uniform_(0, 1)

        e0 = self.model.encode(obs_t)

        adj_mask_1 = self._make_adj_mask(
            np.expand_dims(positions, axis=0)
            if positions is not None
            else None
        )

        if t0:
            self._prev_mean.zero_()
        else:
            self._prev_mean[:, :-1] = self._prev_mean[:, 1:].clone()

            if self.cfg.horizon >= 2:
                self._prev_mean[:, -1] = self._prev_mean[:, -2]

            if horizon >= 2:
                self._prev_mean[:, horizon - 1] = (
                    self._prev_mean[:, horizon - 2]
                )

        num_pi_trajs = int(
            self.cfg.mixture_coef * self.cfg.num_samples
        )
        total_samples = self.cfg.num_samples + num_pi_trajs
        N, H, A = self.N, horizon, self.cfg.action_dim
        B = total_samples

        mean = self._prev_mean[:, :H].clone()
        std = 2 * torch.ones(N, H, A, device=self.device)

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        ):
            for _ in range(self.cfg.iterations):
                noise = torch.randn(
                    H,
                    N,
                    self.cfg.num_samples,
                    A,
                    device=self.device,
                )
                a_random = torch.clamp(
                    mean.permute(1, 0, 2).unsqueeze(2)
                    + std.permute(1, 0, 2).unsqueeze(2) * noise,
                    0,
                    1,
                )

                if num_pi_trajs > 0:
                    S = num_pi_trajs
                    a_pi = torch.empty(
                        H,
                        N,
                        S,
                        A,
                        device=self.device,
                    )

                    e_views = (
                        e0.unsqueeze(0)
                        .expand(N, S, N, -1)
                        .clone()
                    )

                    if adj_mask_1 is not None:
                        view_masks_pi = (
                            self._make_view_masks(adj_mask_1)
                            .expand(N, S, N, N)
                        )
                    else:
                        view_masks_pi = None

                    idx = torch.arange(N, device=self.device)

                    for t in range(H):
                        # 各エージェントの暫定行動系列を使って方策入力を作る
                        a_guess = (
                            self._prev_mean[:, t]
                            .unsqueeze(0)
                            .expand(S, N, A)
                        )

                        z_views = self.model.communicate_per_agent(
                            e_views,
                            a_guess,
                            adj_mask_per_agent=view_masks_pi,
                        )

                        pi_views = self.model.pi(z_views, self.std)

                        # view n での、対象エージェント n の方策出力
                        # shape: [N, S, A]
                        a_self = pi_views[idx, :, idx, :]
                        a_pi[t] = a_self

                        # 各view用のjoint actionを、まず全員の暫定行動で埋める。
                        # shape: [N, S, N, A]
                        #   第1軸: どのエージェントのviewか
                        #   第2軸: 軌道サンプル
                        #   第3軸: joint action中のエージェント
                        a_roll_views = (
                            self._prev_mean[:, t]
                            .view(1, 1, N, A)
                            .expand(N, S, N, A)
                            .clone()
                        )

                        # view n では、エージェント n の行動だけをactor出力に置き換える。
                        # それ以外のエージェントは _prev_mean[:, t] のまま。
                        a_roll_views[idx, :, idx, :] = a_self

                        z_next = self.model.communicate_per_agent(
                            e_views,
                            a_roll_views,
                            adj_mask_per_agent=view_masks_pi,
                        )

                        z_flat = z_next.reshape(N * S, N, -1)
                        a_flat = a_roll_views.reshape(N * S, N, A)

                        next_e_flat, _ = self.model.next(
                            z_flat,
                            a_flat,
                        )
                        e_views = next_e_flat.reshape(
                            N,
                            S,
                            N,
                            -1,
                        )

                    actions_all = torch.cat(
                        [a_random, a_pi],
                        dim=2,
                    )
                else:
                    actions_all = a_random

                joint_base = (
                    self._prev_mean[:, :H]
                    .permute(1, 0, 2)
                    .unsqueeze(1)
                    .expand(H, B, N, A)
                )
                joint_all = (
                    joint_base.unsqueeze(0)
                    .expand(N, H, B, N, A)
                    .clone()
                )
                agent_idx_t = torch.arange(
                    N,
                    device=self.device,
                )
                joint_all[
                    agent_idx_t, :, :, agent_idx_t, :
                ] = actions_all.permute(1, 0, 2, 3)

                e_all = e0.repeat(N * B, 1, 1)
                joint_flat = (
                    joint_all
                    .permute(1, 0, 2, 3, 4)
                    .reshape(H, N * B, N, A)
                )

                if adj_mask_1 is not None:
                    view_masks_1 = self._make_view_masks(
                        adj_mask_1
                    )
                    adj_mask_NB = (
                        view_masks_1
                        .expand(N, B, N, N)
                        .reshape(N * B, N, N)
                    )
                else:
                    adj_mask_NB = None

                values_flat = self.estimate_value(
                    e_all,
                    joint_flat,
                    horizon,
                    adj_mask=adj_mask_NB,
                )
                values = torch.stack(
                    [
                        values_flat[n * B:(n + 1) * B, n]
                        for n in range(N)
                    ],
                    dim=0,
                )

                elite_idxs = torch.topk(
                    values,
                    self.cfg.num_elites,
                    dim=1,
                ).indices
                elite_actions = torch.stack(
                    [
                        actions_all[:, n, elite_idxs[n], :]
                        for n in range(N)
                    ],
                    dim=0,
                )
                elite_value = torch.stack(
                    [
                        values[n, elite_idxs[n]]
                        for n in range(N)
                    ],
                    dim=0,
                ).float()

                max_value = elite_value.max(
                    dim=1,
                    keepdim=True,
                ).values
                score = torch.exp(
                    self.cfg.temperature
                    * (elite_value - max_value)
                )
                score = score / (
                    score.sum(dim=1, keepdim=True) + 1e-8
                )

                w = score.unsqueeze(1).unsqueeze(-1)
                _mean = (w * elite_actions).sum(dim=2)
                _std = torch.sqrt(
                    (
                        w
                        * (
                            elite_actions
                            - _mean.unsqueeze(2)
                        ) ** 2
                    ).sum(dim=2)
                ).clamp_(self.std, 2)

                mean = (
                    self.cfg.momentum * mean
                    + (1 - self.cfg.momentum) * _mean
                )
                std = _std

        self._prev_mean[:, :H] = mean
        score_np = score.float().cpu().numpy()

        final_actions = torch.zeros(
            N,
            A,
            device=self.device,
        )
        for n in range(N):
            best_idx = np.random.choice(
                self.cfg.num_elites,
                p=score_np[n],
            )
            a = elite_actions[n, 0, best_idx]
            if not eval_mode:
                a = a + std[n, 0] * torch.randn(
                    A,
                    device=self.device,
                )
            final_actions[n] = a.clamp(0, 1)

        return final_actions

    def update(self, replay_buffer, step):
        """価値・方策は実観測 latent、モデル損失は予測 latent で学習。"""
        beta = h.linear_schedule(self.cfg.per_beta, step)
        obs, next_obses, action, reward, positions, idxs, weights = (
            replay_buffer.sample(beta)
        )

        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(
            self.cfg.std_schedule,
            step,
        )
        self.model.train()

        N = self.N
        B = obs.shape[0]
        H = self.cfg.horizon
        agent_idx = torch.arange(N, device=self.device)

        train_joint_value = self.joint_value_coef != 0.0
        train_individual_value = self.individual_value_coef != 0.0
        log_values = self.diagnostics is not None

        with torch.no_grad():
            value_adj = [
                self._make_adj_mask(positions[t])
                for t in range(H + 1)
            ]
            rollout_adj = value_adj[0]

            value_view_adj = [
                self._make_view_masks(mask)
                if mask is not None
                else None
                for mask in value_adj
            ]
            rollout_view_adj = value_view_adj[0]

        if (
            rollout_adj is not None
            and train_individual_value
            and getattr(
                self.cfg,
                "reward_type",
                "individual",
            ) != "individual"
        ):
            raise ValueError(
                "局所通信で individual_value_coef != 0 とする場合は、"
                "reward_type='individual' が必要です"
            )

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        ):
            e0 = self.model.encode(obs)
            actor_states = []

            consistency_loss = torch.zeros(
                B,
                device=self.device,
            )
            reward_loss = torch.zeros(
                B,
                device=self.device,
            )
            joint_value_loss = torch.zeros(
                B,
                device=self.device,
            )
            individual_value_loss = torch.zeros(
                B,
                device=self.device,
            )
            priority_loss = torch.zeros(
                B,
                device=self.device,
            )

            # ============================================================
            # 全体通信
            # ============================================================
            if rollout_adj is None:
                e_roll = e0

                for t in range(H):
                    current_obs = (
                        obs
                        if t == 0
                        else next_obses[t - 1]
                    )
                    e_real = (
                        e0
                        if t == 0
                        else self.model.encode(current_obs)
                    )
                    actor_states.append(e_real.detach())

                    # 個別 Q だけの診断時も Q は計算する。
                    if train_joint_value or log_values:
                        z_real = self.model.communicate(
                            e_real,
                            action[t],
                            adj_mask=None,
                        )
                        q1, q2 = self.model.Q(
                            z_real,
                            action[t],
                        )

                        if train_joint_value:
                            q_joint1, q_joint2 = (
                                self.model.Q_joint(
                                    q1,
                                    q2,
                                    e_real,
                                )
                            )

                    z_roll = self.model.communicate(
                        e_roll,
                        action[t],
                        adj_mask=None,
                    )
                    e_roll, reward_pred = self.model.next(
                        z_roll,
                        action[t],
                    )

                    with torch.no_grad():
                        next_e_target = (
                            self.model_target.encode(
                                next_obses[t]
                            )
                        )

                        if train_joint_value or log_values:
                            next_z_for_pi = (
                                self.model.communicate(
                                    next_e_target,
                                    action[t + 1],
                                    adj_mask=None,
                                )
                            )
                            next_a = self.model.pi(
                                next_z_for_pi,
                                self.cfg.min_std,
                            )

                            next_z_for_q = (
                                self.model_target.communicate(
                                    next_e_target,
                                    next_a,
                                    adj_mask=None,
                                )
                            )
                            nq1, nq2 = self.model_target.Q(
                                next_z_for_q,
                                next_a,
                            )

                            if log_values:
                                individual_td_target = (
                                    reward[t]
                                    + self.cfg.discount
                                    * torch.min(nq1, nq2)
                                )

                            if train_joint_value:
                                nq_joint1, nq_joint2 = (
                                    self.model_target.Q_joint(
                                        nq1,
                                        nq2,
                                        next_e_target,
                                    )
                                )
                                nq_joint = torch.min(
                                    nq_joint1,
                                    nq_joint2,
                                )

                                joint_reward = (
                                    reward[t][:, :1]
                                    if getattr(
                                        self.cfg,
                                        "reward_type",
                                        "individual",
                                    ) == "global"
                                    else reward[t].sum(
                                        dim=1,
                                        keepdim=True,
                                    )
                                )
                                joint_td_target = (
                                    joint_reward
                                    + self.cfg.discount
                                    * nq_joint
                                )

                    if log_values:
                        self.diagnostics.values(
                            t,
                            q1=q1,
                            q2=q2,
                            individual_td_target=individual_td_target,
                            q_joint1=(
                                q_joint1
                                if train_joint_value
                                else None
                            ),
                            q_joint2=(
                                q_joint2
                                if train_joint_value
                                else None
                            ),
                            joint_td_target=(
                                joint_td_target
                                if train_joint_value
                                else None
                            ),
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

                    if train_joint_value:
                        joint_value_loss += rho * (
                            h.mse(
                                q_joint1.float(),
                                joint_td_target.float(),
                                reduce=False,
                            )
                            + h.mse(
                                q_joint2.float(),
                                joint_td_target.float(),
                                reduce=False,
                            )
                        ).squeeze(-1)

                        priority_loss += rho * (
                            h.l1(
                                q_joint1,
                                joint_td_target,
                                reduce=False,
                            )
                            + h.l1(
                                q_joint2,
                                joint_td_target,
                                reduce=False,
                            )
                        ).squeeze(-1)

            # ============================================================
            # 局所通信
            # ============================================================
            else:
                e_roll_views = (
                    e0.unsqueeze(0)
                    .expand(N, -1, -1, -1)
                    .contiguous()
                )

                neighbor_mask = rollout_adj.permute(
                    1,
                    0,
                    2,
                ).float()
                neighbor_count = (
                    neighbor_mask.sum(dim=2)
                    .clamp(min=1.0)
                )

                self_mask = torch.eye(
                    N,
                    dtype=neighbor_mask.dtype,
                    device=neighbor_mask.device,
                ).unsqueeze(1)

                other_neighbor_mask = (
                    neighbor_mask * (1.0 - self_mask)
                )
                other_neighbor_count = (
                    other_neighbor_mask.sum(dim=2)
                    .clamp(min=1.0)
                )

                neighbor_consistency_coef = float(
                    getattr(
                        self.cfg,
                        "neighbor_consistency_coef",
                        0.5,
                    )
                )
                reward_type = getattr(
                    self.cfg,
                    "reward_type",
                    "individual",
                )

                for t in range(H):
                    current_obs = (
                        obs
                        if t == 0
                        else next_obses[t - 1]
                    )
                    e_real = (
                        e0
                        if t == 0
                        else self.model.encode(current_obs)
                    )
                    e_real_views = (
                        e_real.unsqueeze(0)
                        .expand(N, -1, -1, -1)
                        .contiguous()
                    )

                    actor_states.append(
                        e_real_views.detach()
                    )

                    if (
                        train_joint_value
                        or train_individual_value
                        or log_values
                    ):
                        z_real_views = (
                            self.model.communicate_per_agent(
                                e_real_views,
                                action[t],
                                adj_mask_per_agent=(
                                    value_view_adj[t]
                                ),
                            )
                        )
                        z_real_self = z_real_views[
                            agent_idx, :, agent_idx, :
                        ].permute(1, 0, 2)

                        q1, q2 = self.model.Q(
                            z_real_self,
                            action[t],
                        )

                        if train_joint_value:
                            q_joint1, q_joint2 = (
                                self.model.Q_joint(
                                    q1,
                                    q2,
                                    e_real,
                                )
                            )

                    z_roll_views = (
                        self.model.communicate_per_agent(
                            e_roll_views,
                            action[t],
                            adj_mask_per_agent=(
                                rollout_view_adj
                            ),
                        )
                    )

                    z_roll_flat = z_roll_views.reshape(
                        N * B,
                        N,
                        -1,
                    )
                    a_roll_flat = (
                        action[t]
                        .unsqueeze(0)
                        .expand(N, -1, -1, -1)
                        .reshape(N * B, N, -1)
                    )

                    next_e_flat, reward_pred_flat = (
                        self.model.next(
                            z_roll_flat,
                            a_roll_flat,
                        )
                    )
                    next_e_roll_views = (
                        next_e_flat.reshape(
                            N,
                            B,
                            N,
                            -1,
                        )
                    )
                    reward_pred_views = (
                        reward_pred_flat.reshape(
                            N,
                            B,
                            N,
                        )
                    )

                    with torch.no_grad():
                        next_e_target = (
                            self.model_target.encode(
                                next_obses[t]
                            )
                        )

                        if (
                            train_joint_value
                            or train_individual_value
                            or log_values
                        ):
                            next_e_target_views = (
                                next_e_target
                                .unsqueeze(0)
                                .expand(N, -1, -1, -1)
                                .contiguous()
                            )

                            next_z_for_pi_views = (
                                self.model_target
                                .communicate_per_agent(
                                    next_e_target_views,
                                    action[t + 1],
                                    adj_mask_per_agent=(
                                        value_view_adj[t + 1]
                                    ),
                                )
                            )
                            next_z_for_pi_self = (
                                next_z_for_pi_views[
                                    agent_idx,
                                    :,
                                    agent_idx,
                                    :,
                                ].permute(1, 0, 2)
                            )

                            next_a = self.model.pi(
                                next_z_for_pi_self,
                                self.cfg.min_std,
                            )

                            next_z_for_q_views = (
                                self.model_target
                                .communicate_per_agent(
                                    next_e_target_views,
                                    next_a,
                                    adj_mask_per_agent=(
                                        value_view_adj[t + 1]
                                    ),
                                )
                            )
                            next_z_for_q_self = (
                                next_z_for_q_views[
                                    agent_idx,
                                    :,
                                    agent_idx,
                                    :,
                                ].permute(1, 0, 2)
                            )

                            nq1, nq2 = (
                                self.model_target.Q(
                                    next_z_for_q_self,
                                    next_a,
                                )
                            )

                            if train_individual_value or log_values:
                                individual_td_target = (
                                    reward[t]
                                    + self.cfg.discount
                                    * torch.min(nq1, nq2)
                                )

                            if train_joint_value:
                                nq_joint1, nq_joint2 = (
                                    self.model_target.Q_joint(
                                        nq1,
                                        nq2,
                                        next_e_target,
                                    )
                                )
                                nq_joint = torch.min(
                                    nq_joint1,
                                    nq_joint2,
                                )

                                joint_reward = (
                                    reward[t][:, :1]
                                    if reward_type == "global"
                                    else reward[t].sum(
                                        dim=1,
                                        keepdim=True,
                                    )
                                )
                                joint_td_target = (
                                    joint_reward
                                    + self.cfg.discount
                                    * nq_joint
                                )

                    if log_values:
                        self.diagnostics.values(
                            t,
                            q1=q1,
                            q2=q2,
                            individual_td_target=individual_td_target,
                            q_joint1=(
                                q_joint1
                                if train_joint_value
                                else None
                            ),
                            q_joint2=(
                                q_joint2
                                if train_joint_value
                                else None
                            ),
                            joint_td_target=(
                                joint_td_target
                                if train_joint_value
                                else None
                            ),
                        )

                    e_roll_views = next_e_roll_views
                    rho = self.cfg.rho ** t

                    consistency_raw = h.mse(
                        next_e_roll_views,
                        next_e_target
                        .unsqueeze(0)
                        .expand(N, -1, -1, -1),
                        reduce=False,
                    ).mean(dim=-1)

                    consistency_self = (
                        consistency_raw * self_mask
                    ).sum(dim=2).mean(dim=0)

                    consistency_other = (
                        (
                            consistency_raw
                            * other_neighbor_mask
                        ).sum(dim=2)
                        / other_neighbor_count
                    ).mean(dim=0)

                    consistency_loss += rho * (
                        consistency_self
                        + neighbor_consistency_coef
                        * consistency_other
                    )

                    reward_target_views = (
                        reward[t]
                        .unsqueeze(0)
                        .expand(N, -1, -1)
                    )
                    reward_raw = h.mse(
                        reward_pred_views,
                        reward_target_views,
                        reduce=False,
                    )

                    if reward_type == "individual":
                        reward_self = (
                            reward_raw * self_mask
                        ).sum(dim=2)
                        reward_loss += (
                            rho * reward_self.mean(dim=0)
                        )
                    else:
                        reward_masked = (
                            (
                                reward_raw
                                * neighbor_mask
                            ).sum(dim=2)
                            / neighbor_count
                        )
                        reward_loss += (
                            rho * reward_masked.mean(dim=0)
                        )

                    if train_joint_value:
                        joint_value_loss += rho * (
                            h.mse(
                                q_joint1.float(),
                                joint_td_target.float(),
                                reduce=False,
                            )
                            + h.mse(
                                q_joint2.float(),
                                joint_td_target.float(),
                                reduce=False,
                            )
                        ).squeeze(-1)

                        priority_loss += rho * (
                            h.l1(
                                q_joint1,
                                joint_td_target,
                                reduce=False,
                            )
                            + h.l1(
                                q_joint2,
                                joint_td_target,
                                reduce=False,
                            )
                        ).squeeze(-1)

                    if train_individual_value:
                        individual_value_loss += rho * (
                            h.mse(
                                q1.float(),
                                individual_td_target.float(),
                                reduce=False,
                            )
                            + h.mse(
                                q2.float(),
                                individual_td_target.float(),
                                reduce=False,
                            )
                        ).mean(dim=1)

                        if not train_joint_value:
                            priority_loss += rho * (
                                h.l1(
                                    q1,
                                    individual_td_target,
                                    reduce=False,
                                )
                                + h.l1(
                                    q2,
                                    individual_td_target,
                                    reduce=False,
                                )
                            ).mean(dim=1)

            total_loss = (
                self.cfg.consistency_coef
                * consistency_loss
                + self.cfg.reward_coef
                * reward_loss
                + self.joint_value_coef
                * joint_value_loss
                + self.individual_value_coef
                * individual_value_loss
            )

            weighted_loss = (
                total_loss * weights
            ).mean()
            weighted_loss.register_hook(
                lambda grad: grad * (1 / H)
            )

        weighted_loss.backward()

        # 指定された５グループのみ、クリッピング前の勾配を記録。
        if self.diagnostics is not None:
            self.diagnostics.gradients(
                self.model,
                self.optim,
            )

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.grad_clip_norm,
        )
        self.optim.step()

        # critic optimizer 更新後のパラメータを記録。
        if self.diagnostics is not None:
            self.diagnostics.weights(
                self.model,
                self.optim,
            )

        has_priority_signal = (
            train_joint_value
            or (
                rollout_adj is not None
                and train_individual_value
            )
        )
        if has_priority_signal:
            p_loss = (
                priority_loss
                .clamp(max=1e4)
                .float()
                .detach()
            )
            safe_p_loss = torch.nan_to_num(
                p_loss,
                nan=1.0,
            )
            replay_buffer.update_priorities(
                idxs,
                safe_p_loss,
            )

        # ============================================================
        # Actor 更新
        # ============================================================
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
                        q1, q2 = self.model.Q(
                            z_eval,
                            a_t,
                        )

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
                            agent_idx,
                            :,
                            agent_idx,
                            :,
                        ]

                        a_self = self.model.pi(
                            z_pre_self,
                            0,
                        )
                        a_t = a_self.permute(1, 0, 2)

                        z_eval_views = (
                            self.model.communicate_per_agent(
                                e_real_views,
                                a_t,
                                adj_mask_per_agent=mask_t,
                            )
                        )
                        z_eval_self = z_eval_views[
                            agent_idx,
                            :,
                            agent_idx,
                            :,
                        ].permute(1, 0, 2)

                        q1, q2 = self.model.Q(
                            z_eval_self,
                            a_t,
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
            "consistency_loss": (
                consistency_loss.mean().detach()
            ),
            "reward_loss": reward_loss.mean().detach(),
            "joint_value_loss": (
                joint_value_loss.mean().detach()
            ),
            "individual_value_loss": (
                individual_value_loss.mean().detach()
            ),
            "pi_loss": pi_loss.detach(),
        }

    def save(self, filepath):
        state = {
            "model": self.model.state_dict(),
            "model_target": self.model_target.state_dict(),
            "optim": self.optim.state_dict(),
            "pi_optim": self.pi_optim.state_dict(),
        }
        torch.save(state, filepath)

    def load(self, filepath):
        checkpoint = torch.load(
            filepath,
            map_location=self.device,
        )
        self.model.load_state_dict(
            checkpoint["model"]
        )
        self.model_target.load_state_dict(
            checkpoint["model_target"]
        )
        self.optim.load_state_dict(
            checkpoint["optim"]
        )
        self.pi_optim.load_state_dict(
            checkpoint["pi_optim"]
        )
        print(
            f"[AsynchMATDMPC] Loaded checkpoints "
            f"from {filepath}"
        )

    def _make_view_masks(self, adj_mask_global):
        """
        adj_mask_global: [B, N, N]
        戻り値: [N, B, N, N]
        """
        if adj_mask_global is None:
            return None

        N = adj_mask_global.size(-1)

        members = adj_mask_global.permute(
            1,
            0,
            2,
        )

        view_masks = (
            adj_mask_global.unsqueeze(0)
            & members.unsqueeze(-1)
            & members.unsqueeze(-2)
        )

        eye = torch.eye(
            N,
            dtype=torch.bool,
            device=adj_mask_global.device,
        )
        view_masks = (
            view_masks
            | eye.view(1, 1, N, N)
        )

        return view_masks