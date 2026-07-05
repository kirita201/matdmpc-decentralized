# envs/vmas_wrapper.py
import torch
import numpy as np
import vmas

class VMASWrapper:
    def __init__(self, cfg, render_mode=None):
        self.cfg = cfg
        self.num_envs = 1  # 既存ループに合わせてバッチ次元を1に固定
        self.device = cfg.device
        self.render_mode = render_mode
        
        # タスクの動的読み込み
        if cfg.task == "navigation":
            from envs.custom_vmas.navigation import NavigationScenario
            scenario = NavigationScenario(
                num_agents=cfg.num_agents,
                reward_type=getattr(cfg, "reward_type", "individual"),
                obs_type=getattr(cfg, "obs_type", "local")
            )
        else:
            scenario = cfg.task
            
        self.env = vmas.make_env(
            scenario=scenario,
            num_envs=self.num_envs,
            device=self.device,
            continuous_actions=True,
            wrapper=None,
            seed=cfg.seed,
            n_agents=cfg.num_agents if isinstance(scenario, str) else None
        )
        
        self.N = len(self.env.agents)
        self.agents = self.env.agents
        
        # obs_shape と action_dim の実測
        sample_obs = self._get_obs()
        self.obs_shape = sample_obs.shape[1:]  # [N, obs_dim] -> (obs_dim,)
        self.action_dim = self.env.action_space[0].shape[0] if hasattr(self.env, "action_space") else 2

    def _get_obs(self):
        # VMASはリストのテンソルを返すので [num_envs, N, obs_dim] にスタックして [N, obs_dim] に次元削減(cpu/numpy)
        obs_list = self.env.get_obs()
        obs_tensor = torch.stack(obs_list, dim=1)
        return obs_tensor[0].cpu().numpy()

    def reset(self):
        self.env.reset()
        return self._get_obs()

    def step(self, actions):
        # actions: [N, action_dim] numpy
        action_list = [torch.tensor(actions[i:i+1], dtype=torch.float32, device=self.device) for i in range(self.N)]
        
        obs_list, reward_list, done_tensor, info_list = self.env.step(action_list)
        
        obs = self._get_obs()
        rewards = torch.stack(reward_list, dim=1)[0].cpu().numpy()  # [N]
        done = done_tensor[0].cpu().numpy().item()
        
        # info生成と位置情報の抽出
        info = {}
        positions = torch.stack([a.state.pos for a in self.env.agents], dim=1) # [num_envs, N, 2]
        info['agent_positions'] = positions[0].cpu().numpy()
        
        # 評価用メトリクスの抽出
        if hasattr(self.env.scenario, 'get_metrics'):
            metrics = self.env.scenario.get_metrics()
            for k, v in metrics.items():
                info[k] = v[0].cpu().numpy()
                
        return obs, rewards, done, info

    def render(self):
        if self.render_mode == "rgb_array":
            frames = self.env.render(mode="rgb_array")
            return frames[0] if isinstance(frames, list) else frames
        return None