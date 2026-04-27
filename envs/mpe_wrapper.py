# envs/mpe_wrapper.py
import numpy as np
from pettingzoo.mpe import simple_spread_v3

class MPEWrapper:
    """Wraps PettingZoo MPE to output stacked numpy arrays for MA-TDMPC"""
    def __init__(self, cfg):
        self.env = simple_spread_v3.env(N=cfg.num_agents, local_ratio=0.5, max_cycles=cfg.episode_length, continuous_actions=True)
        self.agents = self.env.possible_agents
        self.N = cfg.num_agents

    def reset(self):
        self.env.reset()
        obs = []
        for agent in self.agents:
            obs.append(self.env.observe(agent))
        return np.stack(obs) # [N, obs_dim]

    def step(self, actions):
        # actions: [N, action_dim] numpy array
        action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}
        
        obs, rewards, dones = [], [], []
        for agent in self.agents:
            self.env.step(action_dict[agent])
            obs.append(self.env.observe(agent))
            rewards.append(self.env.rewards[agent])
            dones.append(self.env.terminations[agent] or self.env.truncations[agent])
            
        return np.stack(obs), np.array(rewards), np.any(dones), {}