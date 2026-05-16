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

        # For planning: track previous mean actions [N, horizon, action_dim]
        self._prev_mean = torch.zeros(self.N, self.cfg.horizon, self.cfg.action_dim, device=self.device)

        self.scaler = torch.amp.GradScaler(device=self.device.type)

    # ──────────────────────────────────────────────────────────
    # comm_range が inf のときは None を返して全結合にフォールバック
    # ──────────────────────────────────────────────────────────
    def _make_adj_mask(self, obs):
        """
        obs: Tensor [..., N, obs_dim]
        Returns BoolTensor [..., N, N] or None (全結合時)
        """
        if self.comm_range == float('inf'):
            return None
        return h.get_comm_graph(obs, self.comm_range)

    @torch.no_grad()
    def estimate_value(self, e0, actions, horizon, adj_mask=None):
        """
        Estimate multi-agent trajectories.

        Parameters
        ----------
        e0       : Tensor [B, N, latent]
        actions  : Tensor [H, B, N, A]
        horizon  : int
        adj_mask : BoolTensor [B, N, N] or None
            ホライズン全体で固定するマスク（plan()の初期フレームで生成）。
            None のとき全結合。

        Returns
        -------
        G : Tensor [B, N]  累積割引報酬
        """
        G, discount = torch.zeros(actions.size(1), self.N, device=self.device), 1.0
        e = e0
        for t in range(horizon):
            z = self.model.communicate(e, actions[t], adj_mask=adj_mask)
            e, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
            
        # Terminal Q value
        last_action = actions[-1]
        z_pre = self.model.communicate(e, last_action, adj_mask=adj_mask)
        pi_a = self.model.pi(z_pre, self.cfg.min_std)
        
        z_eval = self.model.communicate(e, pi_a, adj_mask=adj_mask)
        
        q1, q2 = self.model.Q(z_eval, pi_a)
        q_min = torch.min(q1, q2)
        G += discount * q_min
        return G

    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, N, obs_dim]
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))

        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.N, self.cfg.action_dim, device=self.device).uniform_(0, 1)

        e0 = self.model.encode(obs_t)  # [1, N, latent]

        # ── 通信グラフを初期観測から生成（ホライズン中固定）──
        # [1, N, N]  (None のとき全結合)
        adj_mask_1 = self._make_adj_mask(obs_t)

        # --- prev_mean の更新 ---
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

        e_batch = e0.repeat(B, 1, 1)  # [B, N, latent]

        # adj_mask を B 次元に展開: [B, N, N] or None
        adj_mask_B = adj_mask_1.expand(B, N, N) if adj_mask_1 is not None else None

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
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
                    # piトラジェクトリ用のマスク: [num_pi_trajs, N, N] or None
                    adj_mask_pi = adj_mask_1.expand(num_pi_trajs, N, N) if adj_mask_1 is not None else None
                    for t in range(H):
                        a_joint = self._prev_mean[:, t].unsqueeze(0).repeat(num_pi_trajs, 1, 1)  # [B_pi, N, A]
                        z_curr = self.model.communicate(e_curr, a_joint, adj_mask=adj_mask_pi)
                        pi_out = self.model.pi(z_curr, self.std)  # [B_pi, N, A]
                        a_pi[t] = pi_out.permute(1, 0, 2)         # [N, B_pi, A]
                        a_joint = pi_out
                        e_curr, _ = self.model.next(
                            self.model.communicate(e_curr, a_joint, adj_mask=adj_mask_pi), a_joint
                        )

                    actions_all = torch.cat([a_random, a_pi], dim=2)  # [H, N, B, A]
                else:
                    actions_all = a_random  # [H, N, B, A]

                # ------------------------------------------------
                # 3. Joint actions を構築: [N, H, B, N, A]
                # ------------------------------------------------
                joint_base = self._prev_mean[:, :H].permute(1, 0, 2).unsqueeze(1).expand(H, B, N, A)
                actions_t = actions_all.permute(0, 2, 1, 3)  # [H, B, N, A]

                joint_all = joint_base.unsqueeze(0).expand(N, H, B, N, A).clone()
                agent_idx_t = torch.arange(N, device=self.device)
                joint_all[agent_idx_t, :, :, agent_idx_t, :] = actions_all.permute(1, 0, 2, 3)
                # joint_all: [N, H, B, N, A]

                # ------------------------------------------------
                # 4. 全エージェント分を一括 evaluate
                #    adj_mask を N*B に展開: [N*B, N, N] or None
                # ------------------------------------------------
                e_all = e0.repeat(N * B, 1, 1)  # [N*B, N, latent]
                joint_flat = joint_all.permute(1, 0, 2, 3, 4).reshape(H, N * B, N, A)

                if adj_mask_1 is not None:
                    adj_mask_NB = adj_mask_1.expand(N * B, N, N)
                else:
                    adj_mask_NB = None

                values_flat = self.estimate_value(e_all, joint_flat, horizon, adj_mask=adj_mask_NB)  # [N*B, N]
                values = torch.stack([
                    values_flat[n * B:(n + 1) * B, n] for n in range(N)
                ], dim=0)  # [N, B]

                # ------------------------------------------------
                # 5. Elite 選択・CEM 更新（全エージェント並列）
                # ------------------------------------------------
                elite_idxs = torch.topk(values, self.cfg.num_elites, dim=1).indices  # [N, num_elites]

                elite_actions = torch.stack([
                    actions_all[:, n, elite_idxs[n], :]
                    for n in range(N)
                ], dim=0)  # [N, H, num_elites, A]

                elite_value = torch.stack([
                    values[n, elite_idxs[n]] for n in range(N)
                ], dim=0)  # [N, num_elites]

                elite_value_f32 = elite_value.float()
                max_value = elite_value_f32.max(dim=1, keepdim=True).values  # [N, 1]

                score = torch.exp(self.cfg.temperature * (elite_value_f32 - max_value))  # [N, num_elites]
                score = score / (score.sum(dim=1, keepdim=True) + 1e-8)

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
        # obs: [B, N, obs_dim]

        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        N = self.N

        # ── 通信グラフをホライズン先頭の obs から生成し、ホライズン中固定 ──
        # 方針②: ホライズン初期ステップの通信グラフで固定。
        #
        # adj_mask_global : [B, N, N]  または None（全結合）
        #   adj_mask_global[b, i, j] = True ⟺ サンプルbでエージェントiとjが通信可能
        #
        # adj_mask_per_agent : [N, B, N, N]  または None
        #   [i, b, :, :] = エージェントiの視点での通信グラフ。
        #   研究メモの設計: [i, b, j, k] = エージェントiの視点でjとkが通信可能か。
        #   ただし実運用上は adj_mask_global[b] をそのまま各 i にブロードキャストで使う
        #   （通信グラフは対称・全エージェント共通の距離から生成されるため）。
        with torch.no_grad():
            adj_mask_global = self._make_adj_mask(obs)  # [B, N, N] or None

        # adj_mask_per_agent: [N, B, N, N] — iごとにi行目のみTrueな行を持つグラフ
        # 研究メモ: adj_mask[i, b, j, k] は「エージェントiの通信グラフ」として
        #           iの近傍のみTrueになる行を持つ
        # → adj_mask_global[b] を i 方向にコピーしたうえで、
        #   各iの行を「iの隣接ノード」のみに制限する（行iだけを残し他行はゼロ化）。
        # ただし communicate_per_agent は e_per_agent[i] = [B, N, latent] 全体に
        # attn_mask を適用するので、[i, b, :, :] はそのままiの通信グラフを使えばよい。
        if adj_mask_global is not None:
            # [B, N, N] -> [N, B, N, N]
            adj_mask_per_agent = adj_mask_global.unsqueeze(0).expand(N, -1, -1, -1)
            # .contiguous() で後の reshape に備える
            adj_mask_per_agent = adj_mask_per_agent.contiguous()
        else:
            adj_mask_per_agent = None

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            # ── エンコード ──
            # e_global: [B, N, latent]  (Mixing用グローバル埋め込み)
            # e_per_agent: [N, B, N, latent]
            #   e_per_agent[i] はエージェントiの視点での全エージェント埋め込み。
            #   現時点では encode は全エージェント共通なので e_global を N回コピーして作る。
            #   将来的にエージェントiが範囲外エージェントをゼロパディングする場合は
            #   ここでマスク処理を加える。
            e_global = self.model.encode(obs)                          # [B, N, latent]
            e_per_agent = e_global.unsqueeze(0).expand(N, -1, -1, -1).contiguous()
            #   e_per_agent: [N, B, N, latent]

            # ── グローバル埋め込みのキャッシュ（policy update用） ──
            # es_global[t] = ステップtでの e_global（detach済み）
            es_global = [e_global.detach()]

            consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0

            for t in range(self.cfg.horizon):
                # ────────────────────────────────────────────────────────
                # 1. エージェントiごとに独立した通信を実行
                #
                # z_per_agent[i, b, j] = エージェントiの通信後のエージェントj埋め込み
                # shape: [N, B, N, latent]
                # ────────────────────────────────────────────────────────
                z_per_agent = self.model.communicate_per_agent(
                    e_per_agent, action[t], adj_mask_per_agent=adj_mask_per_agent
                )  # [N, B, N, latent]

                # ── 自己予測分を抽出: z_self[i, b] = z_per_agent[i, b, i] ──
                # shape: [N, B, latent] → transpose → [B, N, latent]
                z_self = torch.stack(
                    [z_per_agent[i, :, i, :] for i in range(N)], dim=1
                )  # [B, N, latent]

                # ── Global Q（CTDE: MixingNetworkへはグローバル e_global を渡す） ──
                q1, q2 = self.model.Q(z_self, action[t])          # [B, N]
                q_joint1, q_joint2 = self.model.Q_joint(q1, q2, e_global)  # [B, 1]

                # ────────────────────────────────────────────────────────
                # 2. 遷移予測: エージェントiごとに z_per_agent[i] で next を計算
                #
                # next_per_agent: [N, B, N, latent]
                # reward_per_agent: [N, B, N]
                #   reward_per_agent[i, b, j] = エージェントiの視点でのjの報酬予測
                # ────────────────────────────────────────────────────────
                # action を [N*B, N, action_dim] に展開して next() へ
                a_flat = action[t].unsqueeze(0).expand(N, -1, -1, -1).reshape(N * self.cfg.batch_size, N, -1)
                z_flat = z_per_agent.reshape(N * self.cfg.batch_size, N, -1)
                next_e_flat, reward_pred_flat = self.model.next(z_flat, a_flat)
                # [N*B, N, latent], [N*B, N]

                B = self.cfg.batch_size
                next_e_per_agent  = next_e_flat.reshape(N, B, N, -1)   # [N, B, N, latent]
                reward_pred_per_agent = reward_pred_flat.reshape(N, B, N)  # [N, B, N]

                # ────────────────────────────────────────────────────────
                # 3. ターゲット計算（no_grad）
                # ────────────────────────────────────────────────────────
                with torch.no_grad():
                    next_obs    = next_obses[t]
                    next_e_global_tgt = self.model_target.encode(next_obs)   # [B, N, latent]

                    # ターゲット側も per_agent 処理
                    next_e_per_agent_tgt = next_e_global_tgt.unsqueeze(0).expand(N, -1, -1, -1).contiguous()

                    next_z_per_agent_tgt = self.model_target.communicate_per_agent(
                        next_e_per_agent_tgt, action[t + 1],
                        adj_mask_per_agent=adj_mask_per_agent
                    )  # [N, B, N, latent]

                    next_z_self_tgt = torch.stack(
                        [next_z_per_agent_tgt[i, :, i, :] for i in range(N)], dim=1
                    )  # [B, N, latent]

                    # pi はグローバル next_z_self_tgt から決定（通信後の自己埋め込みを使用）
                    # ターゲット側の policy 行動を計算: まず communicate 経由で z を求める
                    next_a_pi = self.model.pi(next_z_self_tgt, self.cfg.min_std)  # [B, N, A]

                    # next_a_pi を使って改めて z を求め Q 計算
                    next_z_self_for_q = self.model_target.communicate_per_agent(
                        next_e_per_agent_tgt, next_a_pi,
                        adj_mask_per_agent=adj_mask_per_agent
                    )  # [N, B, N, latent]
                    next_z_self_q = torch.stack(
                        [next_z_self_for_q[i, :, i, :] for i in range(N)], dim=1
                    )  # [B, N, latent]

                    nq1, nq2 = self.model_target.Q(next_z_self_q, next_a_pi)   # [B, N]
                    nq_joint1, nq_joint2 = self.model_target.Q_joint(
                        nq1, nq2, next_e_global_tgt
                    )  # [B, 1]
                    nq_joint = torch.min(nq_joint1, nq_joint2)

                    joint_reward = reward[t].sum(dim=1, keepdim=True)  # [B, 1]
                    td_target    = joint_reward + self.cfg.discount * nq_joint

                # ── e_global を次ステップへ引き継ぐ（z_self からの遷移で更新） ──
                # グローバル遷移: 全エージェント一括で dynamics を回す（Mixing用）
                z_global = self.model.communicate(e_global, action[t], adj_mask=adj_mask_global)
                e_global, _ = self.model.next(z_global, action[t])
                es_global.append(e_global.detach())

                # e_per_agent を next_e_per_agent で更新
                e_per_agent = next_e_per_agent.detach()

                # ────────────────────────────────────────────────────────
                # 4. 損失計算
                # ────────────────────────────────────────────────────────
                rho = self.cfg.rho ** t

                # ── consistency loss ──────────────────────────────────────
                # エージェントiの視点での遷移予測誤差は、
                # iの通信範囲内にいるエージェントj（adj_mask_global[b, i, j]==True）
                # の埋め込みのみを損失に含める。
                #
                # next_e_per_agent[i, b, j] vs next_e_global_tgt[b, j] を比較
                # consistency_raw_per: [N, B, N, latent]
                consistency_raw_per = h.mse(
                    next_e_per_agent,                              # [N, B, N, latent]
                    next_e_global_tgt.unsqueeze(0).expand(N, -1, -1, -1),  # [N, B, N, latent]
                    reduce=False
                )  # [N, B, N, latent]

                if adj_mask_per_agent is not None:
                    # neighbor_mask[i, b, j] = エージェントiがjと通信可能か
                    # adj_mask_per_agent: [N, B, N, N] → iの行iだけ見ればよい
                    # → adj_mask_global[b, i, j] を [N, B, N] に変換
                    neighbor_mask = adj_mask_global.permute(1, 0, 2).float()  # [N, B, N]
                    neighbor_count = neighbor_mask.sum(dim=2, keepdim=True).clamp(min=1.0)  # [N, B, 1]

                    # latent 次元は平均してから [N, B, N] に
                    c_raw_mean = consistency_raw_per.mean(dim=-1)   # [N, B, N]
                    # 通信範囲内 j のみの加重平均 → [N, B]
                    c_masked = (c_raw_mean * neighbor_mask).sum(dim=2) / neighbor_count.squeeze(-1)
                    # エージェント軸（N）を平均して [B]
                    consistency_loss += rho * c_masked.mean(dim=0)
                else:
                    consistency_loss += rho * consistency_raw_per.mean(dim=(0, 2, 3))

                # ── reward loss ───────────────────────────────────────────
                # reward_pred_per_agent[i, b, j]: エージェントiの視点でのjの報酬予測
                # reward[t]: [B, N]  → [N, B, N] に展開して比較
                reward_target = reward[t].unsqueeze(0).expand(N, -1, -1)  # [N, B, N]
                reward_raw_per = h.mse(reward_pred_per_agent, reward_target, reduce=False)  # [N, B, N]

                if adj_mask_per_agent is not None:
                    r_masked = (reward_raw_per * neighbor_mask).sum(dim=2) / neighbor_count.squeeze(-1)
                    reward_loss += rho * r_masked.mean(dim=0)   # [B]
                else:
                    reward_loss += rho * reward_raw_per.mean(dim=(0, 2))

                # ── value / priority loss（Global Q） ────────────────────
                value_loss    += rho * (
                    h.mse(q_joint1, td_target, reduce=False) +
                    h.mse(q_joint2, td_target, reduce=False)
                ).squeeze(-1)
                priority_loss += rho * (
                    h.l1(q_joint1, td_target, reduce=False) +
                    h.l1(q_joint2, td_target, reduce=False)
                ).squeeze(-1)

            total_loss = (
                self.cfg.consistency_coef * consistency_loss +
                self.cfg.reward_coef      * reward_loss +
                self.cfg.value_coef       * value_loss
            )  # [B]

            weighted_loss = (total_loss * weights).mean()
            weighted_loss.register_hook(lambda grad: grad * (1 / self.cfg.horizon))

        weighted_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        p_loss = priority_loss.clamp(max=1e4).float().detach()
        safe_p_loss = torch.nan_to_num(p_loss, nan=1.0)
        replay_buffer.update_priorities(idxs, safe_p_loss)

        # ── Policy Update ──
        self.pi_optim.zero_grad()
        self.model.track_q_grad(False)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            pi_loss = 0
            for t in range(self.cfg.horizon):
                # policy update はグローバル埋め込みで行う（CTDE: 訓練時はグローバル情報を使用可）
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
        print(f"[MATDMPC] Loaded model checkpoints from {filepath}")