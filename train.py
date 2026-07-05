# train.py (修正版)
import os
import torch
import numpy as np
import random
import imageio
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from envs.make_env import make_env
from algorithm.ma_tdmpc import MATDMPC
from algorithm.buffer import ReplayBuffer
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark         = True

class MockConfig:
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
                    # MPE/VMASの違いを吸収 (rgb_array)
                    if isinstance(frame, list):
                        frames.extend(frame)
                    else:
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

    ckpt_dir = Path(getattr(cfg, "ckpt_dir", "checkpoints")) / f"{task_name}_N{cfg.num_agents}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_ckpt_path = ckpt_dir / "model_latest.pt"
    buffer_ckpt_path = ckpt_dir / "buffer_latest.pt"
    state_ckpt_path = ckpt_dir / "state_latest.pt"
    
    # ── MPE/VMAS 次元数の動的取得 ──
    temp_env = make_env(cfg)
    cfg.obs_shape = list(temp_env.obs_shape)
    cfg.action_dim = temp_env.action_dim
    if hasattr(temp_env, "env") and hasattr(temp_env.env, "close"):
        temp_env.env.close()

    env = make_env(cfg)
    eval_env = make_env(cfg, render_mode="rgb_array")

    agent = MATDMPC(cfg)
    buffer = ReplayBuffer(cfg)
    
    start_step = 0
    episode_idx = 0

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

    initial_random_ratio = 0.8    #predator prey カリキュラム学習用
    decay_steps = cfg.train_steps * 0.3

    for step in range(start_step, cfg.train_steps + cfg.episode_length, cfg.episode_length):

        decay_ratio = max(0.0, 1.0 - (step / decay_steps))
        current_ratio = initial_random_ratio * decay_ratio
        
        env.random_ratio = current_ratio
        eval_env.random_ratio = current_ratio

        obs = env.reset()
        done = False
        t = 0
        ep_reward = 0
        
        # 指標ロギング用変数
        ep_goals = []
        ep_cols = []
        ep_pred_catches = 0
        
        while not done:
            action = agent.plan(obs, step=step, t0=(t==0))
            next_obs, reward, done, info = env.step(action.cpu().numpy())
            
            # ※ info['agent_positions'] は将来の分散通信で使用するため、今はモデル・バッファに渡さず破棄
            buffer.add(obs, action.cpu().numpy(), reward, done)
            obs = next_obs
            ep_reward += np.sum(reward)
            
            # メトリクス収集
            if "goals_occupied_now" in info:
                ep_goals.append(info["goals_occupied_now"])
            if "collisions_now" in info:
                ep_cols.append(info["collisions_now"])
            if "predator_catch" in info:
                ep_pred_catches += info["predator_catch"]
                
            t += 1

        writer.add_scalar("Train/EpisodeReward", ep_reward, step)
        
        # タスク固有のロギング
        if ep_goals:
            writer.add_scalar("Train/GoalsOccupied_End", ep_goals[-1], step)
            writer.add_scalar("Train/Collisions_Sum", np.sum(ep_cols), step)
        if ep_pred_catches > 0 or "predator" in task_name:
            writer.add_scalar("Train/PredatorCatches_Sum", ep_pred_catches, step)

        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            update_iterator = tqdm(range(num_updates), desc="Initial Updates") if num_updates > cfg.episode_length else range(num_updates)

            # ─── 【追加】各損失を蓄積するための辞書 ───
            loss_accum = {}

            for i in update_iterator:
                if step == cfg.seed_steps:
                    update_step = i
                else:
                    update_step = step - cfg.episode_length + i

                loss_info = agent.update(buffer, update_step)
                
                # ─── 【追加】各イテレーションの損失を足し合わせる ───
                for loss_name, loss_val in loss_info.items():
                    loss_accum[loss_name] = loss_accum.get(loss_name, 0.0) + loss_val.item()

            # ─── 【修正】ループ終了後、平均値を計算して TensorBoard に一括プロット ───
            for loss_name, loss_sum in loss_accum.items():
                loss_avg = loss_sum / num_updates
                writer.add_scalar(f"Loss/{loss_name}", loss_avg, step)

        episode_idx += 1
        if episode_idx % 5 == 0:
            print(f"Step: {step}, Episode: {episode_idx}, Reward: {ep_reward:.3f}")

        if step > start_step and step % getattr(cfg, "save_freq", 50000) == 0:
            print(f">>> Saving checkpoints at Step {step}...")
            agent.save(model_ckpt_path)
            buffer.save(buffer_ckpt_path)
            torch.save({'step': step, 'episode_idx': episode_idx}, state_ckpt_path)

        if step % cfg.eval_freq == 0 and step > 0:
            eval_reward = evaluate(eval_env, agent, cfg.eval_episodes, step, log_dir ,save_gif=True)
            print(f">>> EVAL at Step {step}: Reward = {eval_reward:.3f}")

    print(">>> Training complete. Saving final checkpoints...")
    agent.save(model_ckpt_path)
    buffer.save(buffer_ckpt_path)
    torch.save({'step': cfg.train_steps, 'episode_idx': episode_idx}, state_ckpt_path)

if __name__ == '__main__':
    train()