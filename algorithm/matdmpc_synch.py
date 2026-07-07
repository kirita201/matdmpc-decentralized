# algorithm/matdmpc_synch.py
import numpy as np
import torch
from copy import deepcopy
import algorithm.helper as h
from algorithm.models import MACLM

class SynchMATDMPC:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        
        # 同期型/全体通信モデルなので comm_range の制約は受けない（常に全体通信）
        self.comm_range = float('inf')

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

        # エージェントごとのアクション分布の平均 [N, horizon, action_dim]
        self._prev_mean = torch.zeros(self.N, self.cfg.horizon, self.cfg.action_dim, device=self.device)

    @torch.no_grad()
    def estimate_value(self, e0, actions, horizon):
        """
        全エージェントのジョイント軌道を一括で評価 (同期型・全体通信)
        actions: [H, B, N, A]
        e0: [B, N, latent]
        """
        G, discount = torch.zeros(actions.size(1), self.N, device=self.device), 1.0
        e = e0
        for t in range(horizon):
            z = self.model.communicate(e, actions[t], adj_mask=None)
            e, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
            
        # 終端Q値の計算
        last_action = actions[-1]
        z_pre = self.model.communicate(e, last_action, adj_mask=None)
        pi_a = self.model.pi(z_pre, self.cfg.min_std)
        z_eval = self.model.communicate(e, pi_a, adj_mask=None)
        
        q1, q2 = self.model.Q(z_eval, pi_a)
        G += discount * torch.min(q1, q2)
        return G

    @torch.no_grad()
    def plan(self, obs, positions=None, eval_mode=False, step=None, t0=True):
        """全エージェント同期プランニング (オリジナル MA-TDMPC 仕様)"""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, N, obs_dim]
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.N, self.cfg.action_dim, device=self.device).uniform_(0, 1)

        e0 = self.model.encode(obs_t)  # [1, N, latent]

        # 軌道シフトによる初期平均の用意
        if t0:
            self._prev_mean.zero_()
        else:
            self._prev_mean[:, :-1] = self._prev_mean[:, 1:].clone()
            z_est = self.model.communicate(e0, self._prev_mean[:, 0].unsqueeze(0), adj_mask=None)
            self._prev_mean[:, -1] = self.model.pi(z_est, 0).squeeze(0)

        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        total_samples = self.cfg.num_samples + num_pi_trajs
        N, H, A = self.N, horizon, self.cfg.action_dim
        B = total_samples

        # ブロードキャストしやすいように [H, N, A] に変換
        mean = self._prev_mean[:, :H].permute(1, 0, 2).clone()
        std  = 2 * torch.ones(H, N, A, device=self.device)
        e_batch = e0.repeat(B, 1, 1)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for i in range(self.cfg.iterations):
                # 1. ジョイントアクションのランダムサンプリング [H, B_random, N, A]
                noise = torch.randn(H, self.cfg.num_samples, N, A, device=self.device)
                a_random = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * noise, 0, 1)

                # 2. Policy サンプルを用いた軌道の生成
                if num_pi_trajs > 0:
                    a_pi = torch.empty(H, num_pi_trajs, N, A, device=self.device)
                    e_curr = e0.repeat(num_pi_trajs, 1, 1)
                    a_joint = mean[0].unsqueeze(0).repeat(num_pi_trajs, 1, 1)
                    for t in range(H):
                        z_curr = self.model.communicate(e_curr, a_joint, adj_mask=None)
                        pi_out = self.model.pi(z_curr, self.std)
                        a_pi[t] = pi_out
                        a_joint = pi_out
                        e_curr, _ = self.model.next(self.model.communicate(e_curr, a_joint, adj_mask=None), a_joint)
                    actions_all = torch.cat([a_random, a_pi], dim=1)
                else:
                    actions_all = a_random

                # 3. ジョイント軌道のバッチ評価 (全エージェント一括で B 個の軌道を評価)
                values = self.estimate_value(e_batch, actions_all, horizon) # 戻り値: [B, N]

                # 4. エージェントごとに独立して CEM 更新
                new_mean = torch.zeros_like(mean)
                new_std = torch.zeros_like(std)
                score_np = np.zeros((N, self.cfg.num_elites))
                elite_actions_all = torch.zeros(N, H, self.cfg.num_elites, A, device=self.device)

                for n in range(N):
                    v_n = values[:, n]
                    elite_idxs = torch.topk(v_n, self.cfg.num_elites).indices
                    
                    # [H, B, N, A] から特定のエージェントの行動のみ抽出
                    elite_a = actions_all[:, elite_idxs, n, :] # [H, num_elites, A]
                    elite_v = v_n[elite_idxs]
                    
                    max_value = elite_v.max()
                    score = torch.exp(self.cfg.temperature * (elite_v - max_value))
                    score = score / (score.sum() + 1e-8)
                    
                    w = score.view(1, -1, 1)
                    _mean = (w * elite_a).sum(dim=1) # [H, A]
                    _std  = torch.sqrt((w * (elite_a - _mean.unsqueeze(1)) ** 2).sum(dim=1)).clamp_(self.std, 2)

                    new_mean[:, n, :] = _mean
                    new_std[:, n, :] = _std
                    score_np[n] = score.float().cpu().numpy()
                    elite_actions_all[n] = elite_a

                mean = self.cfg.momentum * mean + (1 - self.cfg.momentum) * new_mean
                std  = new_std

        # 次回の初期値用に状態を保存 [N, H, A]
        self._prev_mean[:, :H] = mean.permute(1, 0, 2)
        
        # 最終行動の決定
        final_actions = torch.zeros(N, A, device=self.device)
        for n in range(N):
            best_idx = np.random.choice(self.cfg.num_elites, p=score_np[n])
            a = elite_actions_all[n, 0, best_idx]
            if not eval_mode:
                a = a + std[0, n] * torch.randn(A, device=self.device)
            final_actions[n] = a.clamp(0, 1)

        return final_actions

    def update(self, replay_buffer, step):
        """バッファからサンプルしてモデルを更新 (全体通信のみ)"""
        beta = h.linear_schedule(self.cfg.per_beta, step)
        obs, next_obses, action, reward, positions, idxs, weights = replay_buffer.sample(beta)

        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            e = self.model.encode(obs)
            es = [e.detach()]

            consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0

            # 同期型モデルは常に全体通信(adj_mask=None)で損失を計算
            for t in range(self.cfg.horizon):
                z = self.model.communicate(e, action[t], adj_mask=None)
                q1, q2 = self.model.Q(z, action[t])
                q_joint1, q_joint2 = self.model.Q_joint(q1, q2, e)
                
                e, reward_pred = self.model.next(z, action[t])
                
                with torch.no_grad():
                    next_obs = next_obses[t]
                    next_e = self.model_target.encode(next_obs)
                    next_a = self.model.pi(self.model.communicate(next_e, action[t+1], adj_mask=None), self.cfg.min_std)
                    next_z = self.model_target.communicate(next_e, next_a, adj_mask=None)
                    nq1, nq2 = self.model_target.Q(next_z, next_a)
                    nq_joint1, nq_joint2 = self.model_target.Q_joint(nq1, nq2, next_e)
                    nq_joint = torch.min(nq_joint1, nq_joint2)
                    
                    joint_reward = reward[t].sum(dim=1, keepdim=True)
                    td_target = joint_reward + self.cfg.discount * nq_joint
                    
                es.append(e.detach())

                rho = (self.cfg.rho ** t)
                consistency_loss += rho * h.mse(e, next_e, reduce=False).mean(dim=(1,2)) 
                reward_loss += rho * h.mse(reward_pred, reward[t], reduce=False).mean(dim=1)
                value_loss += rho * (h.mse(q_joint1, td_target, reduce=False) + h.mse(q_joint2, td_target, reduce=False)).squeeze(-1)
                priority_loss += rho * (h.l1(q_joint1, td_target, reduce=False) + h.l1(q_joint2, td_target, reduce=False)).squeeze(-1)

            total_loss = (
                self.cfg.consistency_coef * consistency_loss +
                self.cfg.reward_coef      * reward_loss +
                self.cfg.value_coef       * value_loss
            )

            weighted_loss = (total_loss * weights).mean()
            weighted_loss.register_hook(lambda grad: grad * (1 / self.cfg.horizon))

        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        p_loss = priority_loss.clamp(max=1e4).float().detach()
        safe_p_loss = torch.nan_to_num(p_loss, nan=1.0)
        replay_buffer.update_priorities(idxs, safe_p_loss)

        # Policy の更新
        self.pi_optim.zero_grad()
        self.model.track_q_grad(False)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            pi_loss = 0
            for t in range(self.cfg.horizon):
                e_t = es[t]
                a_t = self.model.pi(self.model.communicate(e_t, action[t], adj_mask=None), 0)
                z_t = self.model.communicate(e_t, a_t, adj_mask=None)
                q1, q2 = self.model.Q(z_t, a_t)
                pi_loss += -torch.min(q1, q2).mean() * (self.cfg.rho ** t)
        
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()

        self.model.track_q_grad(True)

        if step % self.cfg.update_freq == 0:
            h.ema(self.model, self.model_target, self.cfg.tau)

        self.model.eval()
        return {
            'total_loss':       total_loss.mean().detach(),
            'consistency_loss': consistency_loss.mean().detach(),
            'reward_loss':      reward_loss.mean().detach(),
            'value_loss':       value_loss.mean().detach(),
            'pi_loss':          pi_loss.detach()
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
        print(f"[SynchMATDMPC] Loaded checkpoints from {filepath}")