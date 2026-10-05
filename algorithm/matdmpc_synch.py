from copy import deepcopy

import numpy as np
import torch

import algorithm.helper as h
from algorithm.models import MACLM


class SynchMATDMPC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.comm_range = float("inf")
        self.diagnostics = None

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

    @torch.no_grad()
    def estimate_value(self, e0, actions, horizon):
        """
        actions: [H, B, N, A]
        e0: [B, N, latent]
        """
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
                adj_mask=None,
            )
            e, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount

        last_action = actions[-1]
        z_pre = self.model.communicate(
            e,
            last_action,
            adj_mask=None,
        )
        pi_a = self.model.pi(z_pre, self.cfg.min_std)
        z_eval = self.model.communicate(
            e,
            pi_a,
            adj_mask=None,
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

        mean = (
            self._prev_mean[:, :H]
            .permute(1, 0, 2)
            .clone()
        )
        std = 2 * torch.ones(
            H,
            N,
            A,
            device=self.device,
        )
        e_batch = e0.repeat(B, 1, 1)

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        ):
            for _ in range(self.cfg.iterations):
                noise = torch.randn(
                    H,
                    self.cfg.num_samples,
                    N,
                    A,
                    device=self.device,
                )
                a_random = torch.clamp(
                    mean.unsqueeze(1)
                    + std.unsqueeze(1) * noise,
                    0,
                    1,
                )

                if num_pi_trajs > 0:
                    a_pi = torch.empty(
                        H,
                        num_pi_trajs,
                        N,
                        A,
                        device=self.device,
                    )
                    e_curr = e0.repeat(
                        num_pi_trajs,
                        1,
                        1,
                    )
                    a_joint = (
                        mean[0]
                        .unsqueeze(0)
                        .repeat(num_pi_trajs, 1, 1)
                    )

                    for t in range(H):
                        z_curr = self.model.communicate(
                            e_curr,
                            a_joint,
                            adj_mask=None,
                        )
                        pi_out = self.model.pi(
                            z_curr,
                            self.std,
                        )
                        a_pi[t] = pi_out
                        a_joint = pi_out

                        z_next = self.model.communicate(
                            e_curr,
                            a_joint,
                            adj_mask=None,
                        )
                        e_curr, _ = self.model.next(
                            z_next,
                            a_joint,
                        )

                    actions_all = torch.cat(
                        [a_random, a_pi],
                        dim=1,
                    )
                else:
                    actions_all = a_random

                values = self.estimate_value(
                    e_batch,
                    actions_all,
                    horizon,
                )

                new_mean = torch.zeros_like(mean)
                new_std = torch.zeros_like(std)
                score_np = np.zeros(
                    (N, self.cfg.num_elites)
                )
                elite_actions_all = torch.zeros(
                    N,
                    H,
                    self.cfg.num_elites,
                    A,
                    device=self.device,
                )

                for n in range(N):
                    v_n = values[:, n]
                    elite_idxs = torch.topk(
                        v_n,
                        self.cfg.num_elites,
                    ).indices

                    elite_a = actions_all[
                        :, elite_idxs, n, :
                    ]
                    elite_v = v_n[elite_idxs]

                    max_value = elite_v.max()
                    score = torch.exp(
                        self.cfg.temperature
                        * (elite_v - max_value)
                    )
                    score = score / (
                        score.sum() + 1e-8
                    )

                    w = score.view(1, -1, 1)
                    _mean = (w * elite_a).sum(dim=1)
                    _std = torch.sqrt(
                        (
                            w
                            * (
                                elite_a
                                - _mean.unsqueeze(1)
                            ) ** 2
                        ).sum(dim=1)
                    ).clamp_(self.std, 2)

                    new_mean[:, n, :] = _mean
                    new_std[:, n, :] = _std
                    score_np[n] = (
                        score.float().cpu().numpy()
                    )
                    elite_actions_all[n] = elite_a

                mean = (
                    self.cfg.momentum * mean
                    + (1 - self.cfg.momentum)
                    * new_mean
                )
                std = new_std

        self._prev_mean[:, :H] = mean.permute(
            1,
            0,
            2,
        )

        final_actions = torch.zeros(
            N,
            A,
            device=self.device,
        )
        score_tensor = torch.tensor(
            score_np,
            dtype=torch.float32,
            device=self.device,
        )

        for n in range(N):
            best_idx = torch.multinomial(
                score_tensor[n],
                num_samples=1,
            ).item()

            a = elite_actions_all[
                n,
                0,
                best_idx,
            ]
            if not eval_mode:
                a = a + std[0, n] * torch.randn(
                    A,
                    device=self.device,
                )
            final_actions[n] = a.clamp(0, 1)

        return final_actions

    def update(self, replay_buffer, step):
        beta = h.linear_schedule(
            self.cfg.per_beta,
            step,
        )
        obs, next_obses, action, reward, positions, idxs, weights = (
            replay_buffer.sample(beta)
        )

        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(
            self.cfg.std_schedule,
            step,
        )
        self.model.train()

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        ):
            e = self.model.encode(obs)
            es = [e.detach()]

            consistency_loss = 0
            reward_loss = 0
            value_loss = 0
            priority_loss = 0

            for t in range(self.cfg.horizon):
                z = self.model.communicate(
                    e,
                    action[t],
                    adj_mask=None,
                )
                q1, q2 = self.model.Q(
                    z,
                    action[t],
                )
                q_joint1, q_joint2 = (
                    self.model.Q_joint(
                        q1,
                        q2,
                        e,
                    )
                )

                e, reward_pred = self.model.next(
                    z,
                    action[t],
                )

                with torch.no_grad():
                    next_e = self.model_target.encode(
                        next_obses[t]
                    )
                    next_z_for_pi = (
                        self.model.communicate(
                            next_e,
                            action[t + 1],
                            adj_mask=None,
                        )
                    )
                    next_a = self.model.pi(
                        next_z_for_pi,
                        self.cfg.min_std,
                    )
                    next_z = (
                        self.model_target.communicate(
                            next_e,
                            next_a,
                            adj_mask=None,
                        )
                    )
                    nq1, nq2 = self.model_target.Q(
                        next_z,
                        next_a,
                    )
                    nq_joint1, nq_joint2 = (
                        self.model_target.Q_joint(
                            nq1,
                            nq2,
                            next_e,
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
                    td_target = (
                        joint_reward
                        + self.cfg.discount
                        * nq_joint
                    )

                    if self.diagnostics is not None:
                        individual_td_target = (
                            reward[t]
                            + self.cfg.discount
                            * torch.min(nq1, nq2)
                        )

                if self.diagnostics is not None:
                    self.diagnostics.values(
                        t,
                        q1=q1,
                        q2=q2,
                        individual_td_target=individual_td_target,
                        q_joint1=q_joint1,
                        q_joint2=q_joint2,
                        joint_td_target=td_target,
                    )

                es.append(e.detach())

                rho = self.cfg.rho ** t
                consistency_loss += rho * h.mse(
                    e,
                    next_e,
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
                    h.l1(
                        q_joint1,
                        td_target,
                        reduce=False,
                    )
                    + h.l1(
                        q_joint2,
                        td_target,
                        reduce=False,
                    )
                ).squeeze(-1)

            total_loss = (
                self.cfg.consistency_coef
                * consistency_loss
                + self.cfg.reward_coef
                * reward_loss
                + self.cfg.value_coef
                * value_loss
            )

            weighted_loss = (
                total_loss * weights
            ).mean()
            weighted_loss.register_hook(
                lambda grad: grad * (
                    1 / self.cfg.horizon
                )
            )

        weighted_loss.backward()

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

        if self.diagnostics is not None:
            self.diagnostics.weights(
                self.model,
                self.optim,
            )

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

        # Actor は更新するが、今回の Grad/Weight には記録しない。
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        try:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
            ):
                pi_loss = 0

                for t in range(self.cfg.horizon):
                    e_t = es[t]
                    z_pre = self.model.communicate(
                        e_t,
                        action[t],
                        adj_mask=None,
                    )
                    a_t = self.model.pi(z_pre, 0)
                    z_t = self.model.communicate(
                        e_t,
                        a_t,
                        adj_mask=None,
                    )
                    q1, q2 = self.model.Q(
                        z_t,
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
            "reward_loss": (
                reward_loss.mean().detach()
            ),
            "value_loss": value_loss.mean().detach(),
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
            f"[SynchMATDMPC] Loaded checkpoints "
            f"from {filepath}"
        )