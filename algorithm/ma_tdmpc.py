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
        self.model = MACLM(cfg).to(self.device)
        self.model_target = deepcopy(self.model)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
        
        self.model.eval()
        self.model_target.eval()
        self.N = cfg.num_agents

        # For planning: track previous mean actions [N, horizon, action_dim]
        self._prev_mean = torch.zeros(self.N, self.cfg.horizon, self.cfg.action_dim, device=self.device)

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
        z = self.model.communicate(e, self.model.pi(e, self.cfg.min_std))
        pi_a = self.model.pi(z, self.cfg.min_std)
        q1, q2 = self.model.Q(z, pi_a)
        q_min = torch.min(q1, q2)
        G += discount * q_min
        return G

    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) # [1, N, obs_dim]
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
        
        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.N, self.cfg.action_dim, device=self.device).uniform_(-1, 1)

        # Encode observation
        e0 = self.model.encode(obs) # [1, N, latent]

        # Shift prev mean to estimate other agents' futures
        if t0:
            self._prev_mean.zero_()
        else:
            self._prev_mean[:, :-1] = self._prev_mean[:, 1:].clone()
            # Fill last step with policy prediction (rough estimate)
            z_est = self.model.communicate(e0, self._prev_mean[:, 0].unsqueeze(0))
            self._prev_mean[:, -1] = self.model.pi(z_est, 0).squeeze(0)

        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        total_samples = self.cfg.num_samples + num_pi_trajs
        
        final_actions = torch.zeros(self.N, self.cfg.action_dim, device=self.device)
        
        # CEM optimization per agent
        for agent_idx in range(self.N):
            mean = self._prev_mean[agent_idx].clone() # [H, A]
            std = 2 * torch.ones(horizon, self.cfg.action_dim, device=self.device)

            e_batch = e0.repeat(total_samples, 1, 1) # [B, N, latent]

            for i in range(self.cfg.iterations):
                # 1. Sample own actions
                a_own_random = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                    torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device), -1, 1)
                
                # Sample pi actions for own
                if num_pi_trajs > 0:
                    a_own_pi = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
                    e_curr = e0.repeat(num_pi_trajs, 1, 1)
                    for t in range(horizon):
                        # Construct joint action to step env forward
                        a_joint = self._prev_mean[:, t].unsqueeze(0).repeat(num_pi_trajs, 1, 1) # [B, N, A]
                        z_curr = self.model.communicate(e_curr, a_joint)
                        a_own_pi[t] = self.model.pi(z_curr[:, agent_idx].unsqueeze(1), self.cfg.min_std).squeeze(1)
                        # Replace agent's action
                        a_joint[:, agent_idx] = a_own_pi[t]
                        e_curr, _ = self.model.next(self.model.communicate(e_curr, a_joint), a_joint)
                        
                    actions_own = torch.cat([a_own_random, a_own_pi], dim=1) # [H, B, A]
                else:
                    actions_own = a_own_random

                # 2. Build joint actions [H, B, N, A] using previous means for OTHER agents
                joint_actions = self._prev_mean.unsqueeze(1).repeat(1, total_samples, 1, 1).transpose(0, 1) # [H, B, N, A]
                joint_actions[:, :, agent_idx, :] = actions_own

                # 3. Evaluate trajectories
                values = self.estimate_value(e_batch, joint_actions, horizon)[:, agent_idx] # [B]
                
                # 4. Select elites and update
                elite_idxs = torch.topk(values, self.cfg.num_elites, dim=0).indices
                elite_value = values[elite_idxs]
                elite_actions = actions_own[:, elite_idxs]

                max_value = elite_value.max(0)[0]
                score = torch.exp(self.cfg.temperature * (elite_value - max_value))
                score /= score.sum(0)
                _mean = torch.sum(score.unsqueeze(0).unsqueeze(-1) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
                _std = torch.sqrt(torch.sum(score.unsqueeze(0).unsqueeze(-1) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
                _std = _std.clamp_(self.std, 2)
                mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

            # Store the updated mean for this agent
            self._prev_mean[agent_idx] = mean
            
            # Select action
            score = score.cpu().numpy()
            best_idx = np.random.choice(np.arange(score.shape[0]), p=score)
            a = elite_actions[0, best_idx]
            if not eval_mode:
                a += std[0] * torch.randn(self.cfg.action_dim, device=self.device)
            final_actions[agent_idx] = a.clamp(-1, 1)

        return final_actions

    def update(self, replay_buffer, step):
        obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

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
                next_a = self.model.pi(self.model.communicate(next_e, action[t+1] if t < self.cfg.horizon-1 else action[t]), self.cfg.min_std)
                next_z = self.model_target.communicate(next_e, next_a)
                nq1, nq2 = self.model_target.Q(next_z, next_a)
                nq_joint1, nq_joint2 = self.model_target.Q_joint(nq1, nq2, next_e)
                nq_joint = torch.min(nq_joint1, nq_joint2)
                
                joint_reward = reward[t].sum(dim=1, keepdim=True) # [B, 1]
                td_target = joint_reward + self.cfg.discount * nq_joint
                
            es.append(e.detach())

            rho = (self.cfg.rho ** t)
            consistency_loss += rho * torch.mean(h.mse(e, next_e), dim=(1,2)).mean()
            reward_loss += rho * h.mse(reward_pred, reward[t])
            value_loss += rho * (h.mse(q_joint1, td_target) + h.mse(q_joint2, td_target))
            priority_loss += rho * (h.l1(q_joint1, td_target) + h.l1(q_joint2, td_target)).squeeze(-1)

        total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
                     self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
                     self.cfg.value_coef * value_loss.clamp(max=1e4)
        
        weighted_loss = (total_loss * weights.mean()) # simplified weighting
        weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
        weighted_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()
        replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

        # Update Policy
        self.pi_optim.zero_grad()
        self.model.track_q_grad(False)
        pi_loss = 0
        for t, e_t in enumerate(es):
            a_t = self.model.pi(self.model.communicate(e_t, action[0]), self.cfg.min_std) # Approx
            z_t = self.model.communicate(e_t, a_t)
            q1, q2 = self.model.Q(z_t, a_t)
            pi_loss += -torch.min(q1, q2).mean() * (self.cfg.rho ** t)
        
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        if step % self.cfg.update_freq == 0:
            h.ema(self.model, self.model_target, self.cfg.tau)

        self.model.eval()
        return {'total_loss': float(total_loss.item())}