# eval_main.py
import argparse
import torch
import numpy as np
import imageio
import yaml
from pathlib import Path

from envs.make_env import make_env
# train.pyに合わせて各アルゴリズムをインポート
from algorithm.matdmpc_asynch import AsynchMATDMPC
from algorithm.matdmpc_synch import SynchMATDMPC
from algorithm.maddpg import MADDPG
from algorithm.mappo import MAPPO

class MockConfig:
    """Mock Config mimicking OmegaConf/yaml load"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_cfg(yaml_path="cfgs/default.yaml"):
    with open(yaml_path, "r") as f:
        return MockConfig(**yaml.safe_load(f))

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_main(args):
    cfg = load_cfg()
    set_seed(args.seed)
    
    # train.pyのフォーマットに合わせてconfigからパラメータを取得
    task_name = getattr(cfg, "task", "simple_spread")
    obs_type = getattr(cfg, "obs_type", "local")
    reward_type = getattr(cfg, "reward_type", "individual")
    algo_type = getattr(cfg, "algo_type", "asynch")
    comm_type = getattr(cfg, "comm_type", "local")
    
    # train.pyと同一の識別用ディレクトリ名を作成
    exp_name = f"{task_name}_N{cfg.num_agents}_obs:{obs_type}_rew:{reward_type}_algo:{algo_type}_comm:{comm_type}"
    
    # 評価用の環境
    env = make_env(cfg, render_mode="rgb_array" if args.save_gif else None)
    
    # モデル初期化前に次元数を動的取得して設定
    cfg.obs_shape = list(env.obs_shape)
    cfg.action_dim = env.action_dim
    
    # train.pyの仕様に基づいてアルゴリズムを初期化
    if algo_type == "asynch":
        agent = AsynchMATDMPC(cfg)
    elif algo_type == "synch":
        agent = SynchMATDMPC(cfg)
    elif algo_type == "maddpg":
        agent = MADDPG(cfg)
    elif algo_type == "mappo":
        agent = MAPPO(cfg)
    else:
        raise ValueError(f"Unknown algo_type: {algo_type}")
    
    # モデルのロード
    ckpt_dir = Path(getattr(cfg, "ckpt_dir", "checkpoints")) / exp_name
    ckpt_path = Path(args.ckpt) if args.ckpt else ckpt_dir / "model_latest.pt"
    
    if ckpt_path.exists():
        print(f">>> Loading checkpoint from {ckpt_path}")
        agent.load(ckpt_path)
    else:
        print(f">>> Warning: Checkpoint not found at {ckpt_path}. Using an untrained random model.")

    # 評価結果の保存先
    out_dir = Path("eval_results") / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_rewards = []
    episode_metrics = {
        "goals_occupied_now": [],
        "collisions_now": [],
        "predator_catch": [],
    }
    print(f"--- Starting Evaluation for {args.episodes} episodes ---")

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        t = 0
        frames = []

        # キーが存在しない環境では、その指標を集計しない
        ep_goals = None
        ep_collisions = None
        ep_catches = None

        while not done:
            if args.save_gif and ep < args.gif_episodes:
                frame = env.render()
                if frame is not None:
                    if isinstance(frame, list):
                        frames.extend(frame)
                    else:
                        frames.append(frame)

            positions = info.get("agent_positions")

            plan_out = agent.plan(
                obs,
                positions=positions,
                eval_mode=True,
                step=1000000,
                t0=(t == 0),
            )
            action = (
                plan_out[0]
                if algo_type == "mappo" and isinstance(plan_out, tuple)
                else plan_out
            )

            obs, reward, done, info = env.step(action.cpu().numpy())

            # train.py の recorded_reward() と同じ扱い
            rewards = np.asarray(reward)
            if reward_type == "global":
                ep_reward += float(rewards.reshape(-1)[0])
            else:
                ep_reward += float(rewards.sum())

            if "goals_occupied_now" in info:
                ep_goals = float(info["goals_occupied_now"])

            if "collisions_now" in info:
                if ep_collisions is None:
                    ep_collisions = 0.0
                ep_collisions += float(info["collisions_now"])

            if "predator_catch" in info:
                if ep_catches is None:
                    ep_catches = 0.0
                ep_catches += float(info["predator_catch"])

            t += 1

        episode_rewards.append(ep_reward)

        metrics_text = []
        for key, value, label in (
            ("goals_occupied_now", ep_goals, "Goals (end)"),
            ("collisions_now", ep_collisions, "Collisions (sum)"),
            ("predator_catch", ep_catches, "Catches (sum)"),
        ):
            if value is not None:
                episode_metrics[key].append(value)
                metrics_text.append(f"{label} = {value:.2f}")

        suffix = f", {', '.join(metrics_text)}" if metrics_text else ""
        print(
            f"Episode {ep + 1:02d}/{args.episodes}: "
            f"Reward = {ep_reward:.2f}{suffix}"
        )

        if args.save_gif and ep < args.gif_episodes and frames:
            gif_path = out_dir / f"main_eval_ep{ep + 1}.gif"
            imageio.mimsave(gif_path, frames, fps=15)
            print(f"  -> Saved GIF: {gif_path}")

    print("=" * 50)
    print(f"Evaluation Complete! [{exp_name}]")
    print(
        f"Mean Reward: {np.mean(episode_rewards):.2f} "
        f"± {np.std(episode_rewards):.2f}"
    )

    for key, label in (
        ("goals_occupied_now", "Mean Goals Occupied (end)"),
        ("collisions_now", "Mean Collisions (sum)"),
        ("predator_catch", "Mean Predator Catches (sum)"),
    ):
        values = episode_metrics[key]
        if values:
            print(
                f"{label}: {np.mean(values):.2f} "
                f"({len(values)}/{args.episodes} episodes)"
            )

    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multi-Agent RL Model")
    parser.add_argument("--episodes", type=int, default=10, help="評価するトータルエピソード数")
    parser.add_argument("--save_gif", action="store_true", help="GIFを保存するかどうか")
    parser.add_argument("--gif_episodes", type=int, default=1, help="GIFとして保存するエピソード数 (先頭からN個)")
    parser.add_argument("--ckpt", type=str, default=None, help="特定のチェックポイントを指定する場合のパス")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    args = parser.parse_args()
    
    evaluate_main(args)


     #python eval_main.py --episodes 10 --save_gif --gif_episodes 5