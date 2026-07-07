# algorithm/tdmpc_asynch.py
import numpy as np
import torch
from copy import deepcopy
import algorithm.helper as h
from algorithm.models import MACLM  # モデルファイルを models.py に統一することを想定

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
        adj_mask_1 = self._make_adj_mask(positions.unsqueeze(0) if positions is not None else None) # [1, N, N] または None

        # --- prev_mean (前回導出解) の更新 ---
        if t0:
            self._prev_mean.zero_()
        else:
            self._prev_mean[:, :-1] = self._prev_mean[:, 1:].clone()
            z_est = self.model.communicate(e0, self._prev_mean[:, 0].unsqueeze(0), adj_mask=adj_mask_1)
            self._prev_mean[:, -1] = self.model.pi(z_est, 0).squeeze(0)

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
                    a_pi = torch.empty(H, N, num_pi_trajs, A, device=self.device)
                    e_curr = e0.repeat(num_pi_trajs, 1, 1)
                    adj_mask_pi = adj_mask_1.expand(num_pi_trajs, N, N) if adj_mask_1 is not None else None
                    for t in range(H):
                        a_joint = self._prev_mean[:, t].unsqueeze(0).repeat(num_pi_trajs, 1, 1)
                        z_curr = self.model.communicate(e_curr, a_joint, adj_mask=adj_mask_pi)
                        pi_out = self.model.pi(z_curr, self.std)
                        a_pi[t] = pi_out.permute(1, 0, 2)
                        a_joint = pi_out
                        e_curr, _ = self.model.next(self.model.communicate(e_curr, a_joint, adj_mask=adj_mask_pi), a_joint)
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
                adj_mask_NB = adj_mask_1.expand(N * B, N, N) if adj_mask_1 is not None else None

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
        """バッファからサンプルしてモデルを更新 (全体/局所通信の最適化分岐を導入)"""
        beta = h.linear_schedule(self.cfg.per_beta, step)
        # 拡張されたバッファから絶対座標 (positions) も受け取る
        obs, next_obses, action, reward, positions, idxs, weights = replay_buffer.sample(beta)

        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        N, B = self.N, self.cfg.batch_size

        # ホライズン初期ステップ(t=0)の位置から通信マスクを生成
        # positions: [H+1, B, N, 2] -> positions[0]: [B, N, 2]
        with torch.no_grad():
            adj_mask_global = self._make_adj_mask(positions[0])  # [B, N, N] または None

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            e_global = self.model.encode(obs)  # [B, N, latent]
            es_global = [e_global.detach()]

            consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0

            # ────────────────────────────────────────────────────────
            # パターンA：全体通信ルート (comm_range == inf のとき一括高速処理)
            # ────────────────────────────────────────────────────────
            if adj_mask_global is None:
                e = e_global
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
                        
                    es_global.append(e.detach())

                    rho = (self.cfg.rho ** t)
                    consistency_loss += rho * h.mse(e, next_e, reduce=False).mean(dim=(1,2)) 
                    reward_loss += rho * h.mse(reward_pred, reward[t], reduce=False).mean(dim=1)
                    value_loss += rho * (h.mse(q_joint1, td_target, reduce=False) + h.mse(q_joint2, td_target, reduce=False)).squeeze(-1)
                    priority_loss += rho * (h.l1(q_joint1, td_target, reduce=False) + h.l1(q_joint2, td_target, reduce=False)).squeeze(-1)

            # ────────────────────────────────────────────────────────
            # パターンB：局所通信ルート (エージェント個別視点に展開してマスク処理)
            # ────────────────────────────────────────────────────────
            else:
                # [B, N, N] -> [N, B, N, N] にブロードキャスト展開
                adj_mask_per_agent = adj_mask_global.unsqueeze(0).expand(N, -1, -1, -1).contiguous()
                e_per_agent = e_global.unsqueeze(0).expand(N, -1, -1, -1).contiguous()  # [N, B, N, latent]

                for t in range(self.cfg.horizon):
                    # 各エージェント独立通信: [N, B, N, latent]
                    z_per_agent = self.model.communicate_per_agent(
                        e_per_agent, action[t], adj_mask_per_agent=adj_mask_per_agent
                    )
                    # 自己予測分の抽出: [B, N, latent]
                    z_self = torch.stack([z_per_agent[i, :, i, :] for i in range(N)], dim=1)

                    # CTDE: MixingNetwork はグローバルな情報を集約
                    q1, q2 = self.model.Q(z_self, action[t])
                    q_joint1, q_joint2 = self.model.Q_joint(q1, q2, e_global)

                    # 遷移予測 (N*B バッチにフラット化して計算)
                    a_flat = action[t].unsqueeze(0).expand(N, -1, -1, -1).reshape(N * B, N, -1)
                    z_flat = z_per_agent.reshape(N * B, N, -1)
                    next_e_flat, reward_pred_flat = self.model.next(z_flat, a_flat)

                    next_e_per_agent = next_e_flat.reshape(N, B, N, -1)
                    reward_pred_per_agent = reward_pred_flat.reshape(N, B, N)

                    # ターゲット計算
                    with torch.no_grad():
                        next_obs = next_obses[t]
                        next_e_global_tgt = self.model_target.encode(next_obs)
                        next_e_per_agent_tgt = next_e_global_tgt.unsqueeze(0).expand(N, -1, -1, -1).contiguous()

                        next_z_per_agent_tgt = self.model_target.communicate_per_agent(
                            next_e_per_agent_tgt, action[t + 1], adj_mask_per_agent=adj_mask_per_agent
                        )
                        next_z_self_tgt = torch.stack([next_z_per_agent_tgt[i, :, i, :] for i in range(N)], dim=1)
                        next_a_pi = self.model.pi(next_z_self_tgt, self.cfg.min_std)

                        next_z_self_for_q = self.model_target.communicate_per_agent(
                            next_e_per_agent_tgt, next_a_pi, adj_mask_per_agent=adj_mask_per_agent
                        )
                        next_z_self_q = torch.stack([next_z_self_for_q[i, :, i, :] for i in range(N)], dim=1)

                        nq1, nq2 = self.model_target.Q(next_z_self_q, next_a_pi)
                        nq_joint1, nq_joint2 = self.model_target.Q_joint(nq1, nq2, next_e_global_tgt)
                        nq_joint = torch.min(nq_joint1, nq_joint2)

                        joint_reward = reward[t].sum(dim=1, keepdim=True)
                        td_target = joint_reward + self.cfg.discount * nq_joint

                    # 次ステップのグローバル状態の更新 (MixingNetworkへの配信用)
                    z_global = self.model.communicate(e_global, action[t], adj_mask=adj_mask_global)
                    e_global, _ = self.model.next(z_global, action[t])
                    es_global.append(e_global.detach())

                    # 個別状態の更新
                    e_per_agent = next_e_per_agent.detach()

                    # ── 局所通信マスクを考慮した損失計算 ──
                    rho = self.cfg.rho ** t
                    neighbor_mask = adj_mask_global.permute(1, 0, 2).float()  # [N, B, N]
                    neighbor_count = neighbor_mask.sum(dim=2, keepdim=True).clamp(min=1.0)

                    # Consistency Loss
                    consistency_raw_per = h.mse(next_e_per_agent, next_e_global_tgt.unsqueeze(0).expand(N, -1, -1, -1), reduce=False).mean(dim=-1)
                    c_masked = (consistency_raw_per * neighbor_mask).sum(dim=2) / neighbor_count.squeeze(-1)
                    consistency_loss += rho * c_masked.mean(dim=0)

                    # Reward Loss
                    reward_target = reward[t].unsqueeze(0).expand(N, -1, -1)
                    reward_raw_per = h.mse(reward_pred_per_agent, reward_target, reduce=False)
                    r_masked = (reward_raw_per * neighbor_mask).sum(dim=2) / neighbor_count.squeeze(-1)
                    reward_loss += rho * r_masked.mean(dim=0)

                    # Value / Priority Loss
                    value_loss += rho * (h.mse(q_joint1, td_target, reduce=False) + h.mse(q_joint2, td_target, reduce=False)).squeeze(-1)
                    priority_loss += rho * (h.l1(q_joint1, td_target, reduce=False) + h.l1(q_joint2, td_target, reduce=False)).squeeze(-1)

            # ────────────────────────────────────────────────────────
            # 共通の最適化処理
            # ────────────────────────────────────────────────────────
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

        # ── Policy の更新 (訓練時はCTDEの特権により全体通信のマスクを使用可能) ──
        self.pi_optim.zero_grad()
        self.model.track_q_grad(False)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            pi_loss = 0
            for t in range(self.cfg.horizon):
                e_t = es_global[t]
                a_t = self.model.pi(self.model.communicate(e_t, action[t], adj_mask=adj_mask_global), 0)
                z_t = self.model.communicate(e_t, a_t, adj_mask=adj_mask_global)
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
        print(f"[AsynchMATDMPC] Loaded checkpoints from {filepath}")