# envs/vmas_wrapper.py
import torch
import numpy as np
import vmas

class VMASWrapper:
    def __init__(self, cfg, render_mode=None):
        self.cfg = cfg
        self.num_envs = 1  
        
        # 変更点1: num_envs=1 の場合は強制的にCPUを使用（GPUとの通信オーバーヘッドを排除）
        self.device = torch.device("cpu") 
        self.render_mode = render_mode
        
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
            max_steps=cfg.episode_length,
            wrapper=None,
            seed=cfg.seed,
            n_agents=cfg.num_agents if isinstance(scenario, str) else None
        )
        
        self.N = len(self.env.agents)
        self.agents = self.env.agents
        
        sample_obs = self.reset()
        self.obs_shape = sample_obs.shape[1:]  
        self.action_dim = self.env.action_space[0].shape[0] if hasattr(self.env, "action_space") else 2

    # 変更点2: データ収集時は環境モデル内での勾配計算を無効化し、メモリと計算時間を節約
    @torch.no_grad()
    def reset(self):
        obs_list = self.env.reset()
        # CPU上のテンソルであれば .numpy() はメモリコピーが発生せず高速（ゼロコピー）
        obs_tensor = torch.stack(obs_list, dim=1)
        return obs_tensor[0].numpy()

    # 変更点2: 同様に step 内の勾配計算を無効化
    @torch.no_grad()
    def step(self, actions):
        # 変更点3: リスト内包表記での個別テンソル生成をやめ、一括変換後に分割（Pythonオーバーヘッド削減）
        # actions は [N, action_dim] の Numpy配列
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        # unsqueeze(1) で [N, 1, action_dim] にし、list() でラップすると
        # [1, action_dim] のテンソルが N 個入ったリストが瞬時に完成します
        action_list = list(actions_tensor.unsqueeze(1))
        
        obs_list, reward_list, done_tensor, info_list = self.env.step(action_list)
        
        # 変更点4: .cpu() の呼び出しを削除（既にCPU上にあるため不要）
        obs = torch.stack(obs_list, dim=1)[0].numpy()
        rewards = torch.stack(reward_list, dim=1)[0].numpy()
        
        # 変更点5: .numpy().item() ではなく直接 .item() を呼ぶ
        done = done_tensor[0].item()
        
        info = {}
        positions = torch.stack([a.state.pos for a in self.env.agents], dim=1) 
        info['agent_positions'] = positions[0].numpy()
        
        if hasattr(self.env.scenario, 'get_metrics'):
            metrics = self.env.scenario.get_metrics()
            for k, v in metrics.items():
                info[k] = v[0].numpy()
                
        return obs, rewards, done, info

    def render(self):
        if self.render_mode == "rgb_array":
            frames = self.env.render(mode="rgb_array")
            return frames[0] if isinstance(frames, list) else frames
        return None