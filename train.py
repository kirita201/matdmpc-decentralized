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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark         = True

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

    # チェックポイント用ディレクトリの準備
    ckpt_dir = Path(getattr(cfg, "ckpt_dir", "checkpoints")) / f"{task_name}_N{cfg.num_agents}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_ckpt_path = ckpt_dir / "model_latest.pt"
    buffer_ckpt_path = ckpt_dir / "buffer_latest.pt"
    state_ckpt_path = ckpt_dir / "state_latest.pt"
    
    env = MPEWrapper(cfg)
    eval_env = MPEWrapper(cfg, render_mode="rgb_array")

    agent = MATDMPC(cfg)
    buffer = ReplayBuffer(cfg)
    
    start_step = 0
    episode_idx = 0

    # Resume処理
    if getattr(cfg, "resume", False):
        if model_ckpt_path.exists() and buffer_ckpt_path.exists() and state_ckpt_path.exists():
            print(">>> Resuming training from checkpoints...")
            agent.load(model_ckpt_path)
            buffer.load(buffer_ckpt_path)
            state = torch.load(state_ckpt_path)
            start_step = state['step']
            episode_idx = state['episode_idx']
        else:
            print(">>> Checkpoints not found. Starting from scratch.")


    # 元々のノイズ値（設定ファイルから取得、デフォルトは0.0）
    base_noise = getattr(cfg, "prey_noise_std", 0.0)
    initial_extra_noise = 1.0  # カリキュラムのために初期に上乗せするノイズ量
    decay_steps = cfg.train_steps * 0.4  # 例: 全体の半分のステップをかけて減衰

    for step in range(start_step, cfg.train_steps + cfg.episode_length, cfg.episode_length):

        # ── カリキュラム：追加ノイズを徐々に減らし、base_noiseに漸近させる ──
        decay_ratio = max(0.0, 1.0 - (step / decay_steps))
        current_noise = base_noise + (initial_extra_noise * decay_ratio)
        # 環境内の PreyPolicy のノイズ値を更新
        if hasattr(env, "_prey_policy"):
            env._prey_policy.noise_std = current_noise

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
                        writer.add_scalar(f"Loss/{loss_name}", loss_val.item(), step + i)

        episode_idx += 1
        if episode_idx % 5 == 0:
            print(f"Step: {step}, Episode: {episode_idx}, Reward: {ep_reward}")

        # チェックポイントの定期保存
        if step > start_step and step % getattr(cfg, "save_freq", 50000) == 0:
            print(f">>> Saving checkpoints at Step {step}...")
            agent.save(model_ckpt_path)
            buffer.save(buffer_ckpt_path)
            torch.save({'step': step, 'episode_idx': episode_idx}, state_ckpt_path)

        if step % cfg.eval_freq == 0 and step > 0:
            eval_reward = evaluate(eval_env, agent, cfg.eval_episodes, step, log_dir ,save_gif=True)
            print(f">>> EVAL at Step {step}: Reward = {eval_reward}")

    # 最終状態の保存
    print(">>> Training complete. Saving final checkpoints...")
    agent.save(model_ckpt_path)
    buffer.save(buffer_ckpt_path)
    torch.save({'step': cfg.train_steps, 'episode_idx': episode_idx}, state_ckpt_path)

if __name__ == '__main__':
    train()