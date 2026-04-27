# train.py
import torch
import numpy as np
import random
from pathlib import Path
from envs.mpe_wrapper import MPEWrapper
from algorithm.ma_tdmpc import MATDMPC
from algorithm.buffer import ReplayBuffer

class MockConfig:
    """Mock Config mimicking OmegaConf/yaml load"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_cfg():
    import yaml
    with open("cfgs/default.yaml", "r") as f:
        return MockConfig(**yaml.safe_load(f))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(env, agent, num_episodes, step):
    episode_rewards = []
    for _ in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        while not done:
            action = agent.plan(obs, eval_mode=True, step=step, t0=(t==0))
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += np.sum(reward)
            t += 1
        episode_rewards.append(ep_reward)
    return np.mean(episode_rewards)

def train():
    cfg = load_cfg()
    set_seed(cfg.seed)
    
    env = MPEWrapper(cfg)
    agent = MATDMPC(cfg)
    buffer = ReplayBuffer(cfg)
    
    episode_idx = 0
    for step in range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length):
        obs = env.reset()
        done = False
        t = 0
        ep_reward = 0
        
        while not done:
            action = agent.plan(obs, step=step, t0=(t==0))
            next_obs, reward, done, _ = env.step(action.cpu().numpy())
            buffer.add(obs, action.cpu().numpy(), reward, done)
            obs = next_obs
            ep_reward += np.sum(reward)
            t += 1

        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            for i in range(num_updates):
                agent.update(buffer, step + i)

        episode_idx += 1
        if episode_idx % 10 == 0:
            print(f"Step: {step}, Episode: {episode_idx}, Reward: {ep_reward}")

        if step % cfg.eval_freq == 0 and step > 0:
            eval_reward = evaluate(env, agent, cfg.eval_episodes, step)
            print(f">>> EVAL at Step {step}: Reward = {eval_reward}")

if __name__ == '__main__':
    train()