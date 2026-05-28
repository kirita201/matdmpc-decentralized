# eval_prey.py
import os
# Google Colab等のヘッドレス環境でPygameのエラーを防ぐおまじない
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
import torch
import numpy as np
import imageio
from pathlib import Path

from envs.mpe_wrapper import MPEWrapper, PREDPREY_CONFIGS
from envs.prey_policy import PreyPolicy

class DummyCfg:
    """MPEWrapperに渡す最小限のConfig"""
    pass

def evaluate_prey(args):
    cfg = DummyCfg()
    cfg.task = "predator_prey"
    cfg.num_agents = args.N
    cfg.episode_length = args.episode_length
    
    # 評価用の環境
    env = MPEWrapper(cfg, render_mode="rgb_array" if args.save_gif else None)
    
    num_good = PREDPREY_CONFIGS[args.N][0]
    all_agents = env.env.possible_agents
    prey_agents = [a for a in all_agents if "adversary" not in a]
    
    env.env.reset()
    prey_obs_sample = env.env.observe(prey_agents[0])
    obs_dim = prey_obs_sample.shape[0]
    action_dim = env.env.action_space(prey_agents[0]).shape[0]

    # チェックポイントパスの動的解決 (args.ckptが指定されていなければNから推定)
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = Path(f"checkpoints/prey_N{args.N}.pt")

    if ckpt_path.exists():
        print(f">>> Loading Prey checkpoint from {ckpt_path}")
        policy = PreyPolicy.load(ckpt_path, device="cpu", noise_std=0.0)
    else:
        print(f">>> Warning: Checkpoint not found at {ckpt_path}. Using an untrained random policy.")
        policy = PreyPolicy.random(num_good, obs_dim, action_dim, device="cpu")

    out_dir = Path("eval_results") / f"prey_N{args.N}"
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_rewards = []
    print(f"--- Starting Prey Evaluation for {args.episodes} episodes ---")

    for ep in range(args.episodes):
        env.env.reset()
        done = False
        ep_reward = 0
        frames = []
        
        while not done:
            if args.save_gif and ep < args.gif_episodes:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            # Preyの行動をポリシーから取得
            prey_obs = np.stack([env.env.observe(a) for a in prey_agents])
            prey_actions = policy.act(prey_obs)

            # Adversaryの行動取得 (最も近いPreyを追尾するヒューリスティック)
            world = env.env.unwrapped.world if hasattr(env.env, "unwrapped") else env.env.world
            prey_idx = 0
            
            for agent_name in all_agents:
                if "adversary" in agent_name:
                    adv_obj = next((a for a in world.agents if a.name == agent_name), None)
                    preys = [a for a in world.agents if not a.adversary]
                    if adv_obj is not None and len(preys) > 0:
                        closest_prey = min(preys, key=lambda p: np.linalg.norm(p.state.p_pos - adv_obj.state.p_pos))
                        delta_pos = closest_prey.state.p_pos - adv_obj.state.p_pos
                        norm = np.linalg.norm(delta_pos)
                        if norm > 1e-5:
                            delta_pos = delta_pos / norm
                        
                        act = np.zeros(5, dtype=np.float32)
                        act[1] = -max(0, delta_pos[0])
                        act[2] = -max(0, -delta_pos[0])
                        act[3] = -max(0, delta_pos[1])
                        act[4] = -max(0, -delta_pos[1])
                    else:
                        act = env.env.action_space(agent_name).sample()
                else:
                    act = prey_actions[prey_idx]
                    prey_idx += 1
                
                env.env.step(act)

            # Prey目線の報酬と完了判定
            prey_rews = [env.env.rewards[a] for a in prey_agents]
            dones = [env.env.terminations[a] or env.env.truncations[a] for a in prey_agents]
            
            ep_reward += np.mean(prey_rews)
            done = bool(np.any(dones))

        episode_rewards.append(ep_reward)
        print(f"Episode {ep+1:02d}/{args.episodes}: Reward = {ep_reward:.2f}")

        # GIF保存
        if args.save_gif and ep < args.gif_episodes and len(frames) > 0:
            gif_path = out_dir / f"prey_eval_ep{ep+1}.gif"
            imageio.mimsave(gif_path, frames, fps=15)
            print(f"  -> Saved GIF: {gif_path}")

    # 結果の集計
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print("=" * 50)
    print(f"Prey Evaluation Complete!")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Prey Agent")
    parser.add_argument("--N", type=int, default=6, help="Adversary数 (3, 6, 15)")
    parser.add_argument("--episodes", type=int, default=10, help="評価するトータルエピソード数")
    parser.add_argument("--episode_length", type=int, default=25, help="1エピソードの長さ")
    parser.add_argument("--save_gif", action="store_true", help="GIFを保存するかどうか")
    parser.add_argument("--gif_episodes", type=int, default=1, help="GIFとして保存するエピソード数 (先頭からN個)")
    # 初期値をNoneにして、スクリプト内で動的に N の数からパスを作るように変更
    parser.add_argument("--ckpt", type=str, default=None, help="Preyのチェックポイントパス")
    args = parser.parse_args()
    
    evaluate_prey(args)

#python eval_prey.py --N 6 --episodes 10 --save_gif --gif_episodes 3