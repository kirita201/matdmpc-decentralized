# train.py
import os
import torch
import numpy as np
import random
import imageio
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from envs.mpe_wrapper import MPEWrapper
from algorithm.ma_tdmpc import MATDMPC
from algorithm.buffer import ReplayBuffer
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

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

def evaluate(env, agent, num_episodes, step, log_dir, save_gif=False):
    episode_rewards = []
    frames = []
    for ep in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        while not done:

            if save_gif and ep == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            action = agent.plan(obs, eval_mode=True, step=step, t0=(t==0))
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += np.sum(reward)
            t += 1
        episode_rewards.append(ep_reward)
    
    if save_gif and len(frames) > 0:
        gif_path = os.path.join(log_dir, f"eval_step_{step}.gif")
        imageio.mimsave(gif_path, frames, fps=15)

    return np.mean(episode_rewards)

def train():
    cfg = load_cfg()
    set_seed(cfg.seed)

    task_name = getattr(cfg, "task", "simple_spread")
    log_dir = Path(f"logs/{task_name}_N{cfg.num_agents}")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    env = MPEWrapper(cfg)
    eval_env = MPEWrapper(cfg, render_mode="rgb_array")

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

        writer.add_scalar("Train/EpisodeReward", ep_reward, step)

        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length

            # seed_stepsの初回大量アップデートの時だけプログレスバーを表示
            update_iterator = tqdm(range(num_updates), desc="Initial Updates") if num_updates > cfg.episode_length else range(num_updates)

            for i in update_iterator:
                loss_info = agent.update(buffer, step + i)

                if i == num_updates - 1:
                    for loss_name, loss_val in loss_info.items():
                        writer.add_scalar(f"Loss/{loss_name}", loss_val, step + i)

        episode_idx += 1
        if episode_idx % 10 == 0:
            print(f"Step: {step}, Episode: {episode_idx}, Reward: {ep_reward}")

        if step % cfg.eval_freq == 0 and step > 0:
            eval_reward = evaluate(env, agent, cfg.eval_episodes, step)
            print(f">>> EVAL at Step {step}: Reward = {eval_reward}")

if __name__ == '__main__':
    train()