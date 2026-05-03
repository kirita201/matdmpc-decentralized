# algorithm/ma_tdmpc.py
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import algorithm.helper as h
from algorithm.model import MACLM

class MATDMPC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        # モデルの初期化後にコンパイルを追加
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

        # For planning: track previous mean actions [N, horizon, action_dim]
        self._prev_mean = torch.zeros(self.N, self.cfg.horizon, self.cfg.action_dim, device=self.device)

        self.scaler = torch.amp.GradScaler(device=self.device.type)

    @torch.no_grad()
    def estimate_value(self, e0, actions, horizon):
        """
        Estimate multi-agent trajectories.
        actions: [H, B, N, A]
        e0: [B, N, latent]
        Returns: [B, N] cumulative rewards
        """
        G, discount = torch.zeros(actions.size(1), self.N, device=self.device), 1.0
        e = e0
        for t in range(horizon):
            z = self.model.communicate(e, actions[t])
            e, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
            
        # Terminal Q value
        # 1. 軌道の最後のアクションを使って仮の z を作り、方策から pi_a をサンプリングする
        last_action = actions[-1]
        z_pre = self.model.communicate(e, last_action)
        pi_a = self.model.pi(z_pre, self.cfg.min_std)
        
        # 2. 出力された pi_a を使って、Q値評価用の正しい z を作り直す
        z_eval = self.model.communicate(e, pi_a)
        
        # 3. 正しいペア (z_eval, pi_a) で Q値を計算する
        q1, q2 = self.model.Q(z_eval, pi_a)
        q_min = torch.min(q1, q2)
        G += discount * q_min
        return G

    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, N, obs_dim]
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.N, self.cfg.action_dim, device=self.device).uniform_(0, 1)

        e0 = self.model.encode(obs)  # [1, N, latent]

        # --- prev_mean の更新 ---
        if t0:
            self._prev_mean.zero_()
        else:
            self._prev_mean[:, :-1] = self._prev_mean[:, 1:].clone()
            z_est = self.model.communicate(e0, self._prev_mean[:, 0].unsqueeze(0))
            self._prev_mean[:, -1] = self.model.pi(z_est, 0).squeeze(0)

        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        total_samples = self.cfg.num_samples + num_pi_trajs
        N, H, A = self.N, horizon, self.cfg.action_dim
        B = total_samples

        # mean: [N, H, A], std: [N, H, A]
        mean = self._prev_mean[:, :H].clone()
        std  = 2 * torch.ones(N, H, A, device=self.device)

        # e0 を [B, N, latent] に展開（全エージェント共通）
        e_batch = e0.repeat(B, 1, 1)  # [B, N, latent]

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
        #with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            for i in range(self.cfg.iterations):
                # ------------------------------------------------
                # 1. ランダムサンプル: [H, N, num_samples, A]
                # ------------------------------------------------
                noise = torch.randn(H, N, self.cfg.num_samples, A, device=self.device)
                a_random = torch.clamp(
                    mean.permute(1, 0, 2).unsqueeze(2) + std.permute(1, 0, 2).unsqueeze(2) * noise,
                    0, 1
                )  # [H, N, num_samples, A]

                # ------------------------------------------------
                # 2. Policy サンプル: [H, N, num_pi_trajs, A]
                # ------------------------------------------------
                if num_pi_trajs > 0:
                    a_pi = torch.empty(H, N, num_pi_trajs, A, device=self.device)
                    e_curr = e0.repeat(num_pi_trajs, 1, 1)  # [num_pi_trajs, N, latent]
                    for t in range(H):
                        # 全エージェントの joint action（prev_mean ベース）
                        a_joint = self._prev_mean[:, t].unsqueeze(0).repeat(num_pi_trajs, 1, 1)  # [B_pi, N, A]
                        z_curr = self.model.communicate(e_curr, a_joint)
                        # 全エージェント同時に pi を取得: [B_pi, N, A]
                        pi_out = self.model.pi(z_curr, self.std)  # [B_pi, N, A]
                        a_pi[t] = pi_out.permute(1, 0, 2)  # [N, B_pi, A]
                        # 全エージェントの action を置き換えて次ステップへ
                        a_joint = pi_out  # [B_pi, N, A]
                        e_curr, _ = self.model.next(
                            self.model.communicate(e_curr, a_joint), a_joint
                        )

                    # [H, N, B, A] に結合
                    actions_all = torch.cat([a_random, a_pi], dim=2)  # [H, N, B, A]
                else:
                    actions_all = a_random  # [H, N, B, A]

                # ------------------------------------------------
                # 3. Joint actions を構築: [H, B, N, A]
                #    各エージェントの「自分のサンプル」を対角に配置し、
                #    他エージェントは prev_mean で埋める
                # ------------------------------------------------
                # prev_mean ベースの joint: [H, B, N, A]
                joint_base = self._prev_mean[:, :H].permute(1, 0, 2).unsqueeze(1).expand(H, B, N, A)
                # actions_all: [H, N, B, A] → [H, B, N, A] に転置してコピー
                actions_t = actions_all.permute(0, 2, 1, 3)  # [H, B, N, A]

                # 全エージェントの joint を一括構築
                # agent n の joint: actions_t の n 列だけ自分のサンプル、残りは joint_base
                # → [N, H, B, N, A] を作り対角マスクで選択
                joint_all = joint_base.unsqueeze(0).expand(N, H, B, N, A).clone()
                # agent n について joint_all[n, :, :, n, :] = actions_all[:, n, :, :]
                agent_idx_t = torch.arange(N, device=self.device)
                joint_all[agent_idx_t, :, :, agent_idx_t, :] = actions_all.permute(1, 0, 2, 3)
                # joint_all: [N, H, B, N, A]

                # ------------------------------------------------
                # 4. 全エージェント分を一括 evaluate
                #    N 個の agent を B 次元にまとめる: [N*B, N, latent]
                # ------------------------------------------------
                e_all = e0.repeat(N * B, 1, 1)  # [N*B, N, latent]
                # joint_actions を [N*B, ...] に reshape: [H, N*B, N, A]
                joint_flat = joint_all.permute(1, 0, 2, 3, 4).reshape(H, N * B, N, A)

                values_flat = self.estimate_value(e_all, joint_flat, horizon)  # [N*B, N]
                # agent n の価値は values_flat[n*B:(n+1)*B, n]
                values = torch.stack([
                    values_flat[n * B:(n + 1) * B, n] for n in range(N)
                ], dim=0)  # [N, B]

                # ------------------------------------------------
                # 5. Elite 選択・CEM 更新（全エージェント並列）
                # ------------------------------------------------
                elite_idxs = torch.topk(values, self.cfg.num_elites, dim=1).indices  # [N, num_elites]

                # elite_actions: [N, H, num_elites, A]
                elite_actions = torch.stack([
                    actions_all[:, n, elite_idxs[n], :]  # [H, num_elites, A]
                    for n in range(N)
                ], dim=0)  # [N, H, num_elites, A]

                elite_value = torch.stack([
                    values[n, elite_idxs[n]] for n in range(N)
                ], dim=0)  # [N, num_elites]

                # --- 修正: 計算精度を保つためにfloat32にキャスト ---
                elite_value_f32 = elite_value.float()
                max_value = elite_value_f32.max(dim=1, keepdim=True).values  # [N, 1]
                #max_value = elite_value.max(dim=1, keepdim=True).values  # [N, 1]

                # float32のまま計算を行う
                score = torch.exp(self.cfg.temperature * (elite_value_f32 - max_value))  # [N, num_elites]
                score = score / (score.sum(dim=1, keepdim=True) + 1e-8)  # [N, num_elites]
                #score = torch.exp(self.cfg.temperature * (elite_value - max_value))  # [N, num_elites]
                #score = score / (score.sum(dim=1, keepdim=True) + 1e-8)  # [N, num_elites]

                # [N, num_elites] -> [N, 1, num_elites, 1] for broadcasting with [N, H, num_elites, A]
                w = score.unsqueeze(1).unsqueeze(-1)
                _mean = (w * elite_actions).sum(dim=2)  # [N, H, A]
                _std  = torch.sqrt((w * (elite_actions - _mean.unsqueeze(2)) ** 2).sum(dim=2)).clamp_(self.std, 2)

                mean = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean
                std  = _std

        # --- 最終アクション選択 ---
        self._prev_mean[:, :H] = mean

        score_np = score.float().cpu().numpy()  # [N, num_elites]
        final_actions = torch.zeros(N, A, device=self.device)
        for n in range(N):
            best_idx = np.random.choice(self.cfg.num_elites, p=score_np[n])
            a = elite_actions[n, 0, best_idx]  # [A]
            if not eval_mode:
                a = a + std[n, 0] * torch.randn(A, device=self.device)
            final_actions[n] = a.clamp(0, 1)

        return final_actions

    def update(self, replay_buffer, step):
        beta = h.linear_schedule(self.cfg.per_beta, step)
        obs, next_obses, action, reward, idxs, weights = replay_buffer.sample(beta)
        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        #with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            e = self.model.encode(obs)
            es = [e.detach()]

            consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
            
            for t in range(self.cfg.horizon):
                z = self.model.communicate(e, action[t])
                q1, q2 = self.model.Q(z, action[t])
                q_joint1, q_joint2 = self.model.Q_joint(q1, q2, e)
                
                e, reward_pred = self.model.next(z, action[t])
                
                with torch.no_grad():
                    next_obs = next_obses[t]
                    next_e = self.model_target.encode(next_obs)
                    
                    # Target Joint Q
                    next_a = self.model.pi(self.model.communicate(next_e, action[t+1]), self.cfg.min_std)
                    next_z = self.model_target.communicate(next_e, next_a)
                    nq1, nq2 = self.model_target.Q(next_z, next_a)
                    nq_joint1, nq_joint2 = self.model_target.Q_joint(nq1, nq2, next_e)
                    nq_joint = torch.min(nq_joint1, nq_joint2)
                    
                    joint_reward = reward[t].sum(dim=1, keepdim=True) # [B, 1]
                    td_target = joint_reward + self.cfg.discount * nq_joint
                    
                es.append(e.detach())

                rho = (self.cfg.rho ** t)
                # 各サンプルのLossを [B] のベクトルとして計算・累積する
                consistency_loss += rho * h.mse(e, next_e, reduce=False).mean(dim=(1,2)) 
                reward_loss += rho * h.mse(reward_pred, reward[t], reduce=False).mean(dim=1)
                value_loss += rho * (h.mse(q_joint1, td_target, reduce=False) + h.mse(q_joint2, td_target, reduce=False)).squeeze(-1)
                priority_loss += rho * (h.l1(q_joint1, td_target, reduce=False) + h.l1(q_joint2, td_target, reduce=False)).squeeze(-1)

            total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
                     self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
                     self.cfg.value_coef * value_loss.clamp(max=1e4) # 形状: [B]
        
            # PERの重みを各サンプルに掛けてからバッチ平均を取る
            weighted_loss = (total_loss * weights).mean()
            weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))


        #weighted_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        #self.optim.step()

        self.scaler.scale(weighted_loss).backward()
        self.scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.scaler.step(self.optim)
        
        # 修正後: NaNが発生した場合は安全な値(1.0)に変換してバッファの破壊を防ぐ
        p_loss = priority_loss.clamp(max=1e4).float().detach()
        safe_p_loss = torch.nan_to_num(p_loss, nan=1.0)
        replay_buffer.update_priorities(idxs, safe_p_loss)

        # Update Policy
        self.pi_optim.zero_grad()
        self.model.track_q_grad(False)

        #with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            pi_loss = 0
            for t in range(self.cfg.horizon):
                e_t = es[t]
                a_t = self.model.pi(self.model.communicate(e_t, action[t]), 0)
                z_t = self.model.communicate(e_t, a_t)
                q1, q2 = self.model.Q(z_t, a_t)
                pi_loss += -torch.min(q1, q2).mean() * (self.cfg.rho ** t)
        
        #pi_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        #self.pi_optim.step()

        self.scaler.scale(pi_loss).backward()
        self.scaler.unscale_(self.pi_optim)
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.scaler.step(self.pi_optim)
        self.scaler.update()

        self.model.track_q_grad(True)

        if step % self.cfg.update_freq == 0:
            h.ema(self.model, self.model_target, self.cfg.tau)

        self.model.eval()
        return {
            'total_loss': total_loss.mean().detach(),
            'consistency_loss': consistency_loss.mean().detach(),
            'reward_loss': reward_loss.mean().detach(),
            'value_loss': value_loss.mean().detach(),
            'pi_loss': pi_loss.detach()
        }