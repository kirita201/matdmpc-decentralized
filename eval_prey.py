# eval_prey.py
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
import torch
import numpy as np
import imageio
from pathlib import Path

from envs.mpe_wrapper import MPEWrapper, PREDPREY_CONFIGS
from envs.prey_policy import PreyPolicy

class DummyCfg:
    pass

def evaluate_prey(args):
    cfg = DummyCfg()
    cfg.task = "predator_prey"
    cfg.num_agents = args.N
    cfg.episode_length = args.episode_length
    
    env = MPEWrapper(cfg, render_mode="rgb_array" if args.save_gif else None)
    
    num_good = PREDPREY_CONFIGS[args.N][0]
    all_agents = env.env.possible_agents
    prey_agents = [a for a in all_agents if "adversary" not in a]
    
    env.env.reset()
    prey_obs_sample = env.env.observe(prey_agents[0])
    obs_dim = prey_obs_sample.shape[0]
    action_dim = env.env.action_space(prey_agents[0]).shape[0]

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(f"checkpoints/prey_N{args.N}.pt")
    if ckpt_path.exists():
        print(f">>> Loading Prey checkpoint from {ckpt_path}")
        policy = PreyPolicy.load(ckpt_path, device="cpu", noise_std=0.0)
    else:
        print(f">>> Warning: Checkpoint not found at {ckpt_path}. Using an untrained random policy.")
        policy = PreyPolicy.random(num_good, obs_dim, action_dim, device="cpu")

    out_dir = Path("eval_results") / f"prey_N{args.N}"
    out_dir.mkdir(parents=True, exist_ok=True)

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

            prey_obs = np.stack([env.env.observe(a) for a in prey_agents])
            prey_actions = policy.act(prey_obs)

            world = env.env.unwrapped.world if hasattr(env.env, "unwrapped") else env.env.world
            prey_idx = 0
            
            # シンプルなループで順次step処理 (順番は完全に保証される)
            for agent_name in all_agents:
                if "adversary" in agent_name:
                    adv_obj = next((a for a in world.agents if a.name == agent_name), None)
                    preys = [a for a in world.agents if not a.adversary]
                    
                    if adv_obj is not None and len(preys) > 0:
                        closest_prey = min(preys, key=lambda p: np.linalg.norm(p.state.p_pos - adv_obj.state.p_pos))
                        delta_pos = closest_prey.state.p_pos - adv_obj.state.p_pos
                        
                        # ご提案の通り、絶対値の大きい方を1とするスケーリングでフルパワー化
                        scale = max(abs(delta_pos[0]), abs(delta_pos[1]))
                        if scale > 1e-5:
                            delta_pos = delta_pos / scale
                        
                        act = np.zeros(5, dtype=np.float32)
                        # 【修正】MPEの正しい行動インデックス仕様に合わせて配置
                        act[1] = max(0, -delta_pos[0])  # 1: Left (-x)
                        act[2] = max(0,  delta_pos[0])  # 2: Right (+x)
                        act[3] = max(0, -delta_pos[1])  # 3: Down (-y)
                        act[4] = max(0,  delta_pos[1])  # 4: Up (+y)
                    else:
                        act = env.env.action_space(agent_name).sample()
                else:
                    act = prey_actions[prey_idx]
                    prey_idx += 1
                
                env.env.step(act)

            prey_rews = [env.env.rewards[a] for a in prey_agents]
            dones = [env.env.terminations[a] or env.env.truncations[a] for a in prey_agents]
            ep_reward += np.mean(prey_rews)
            done = bool(np.any(dones))

        print(f"Episode {ep+1:02d}/{args.episodes}: Reward = {ep_reward:.2f}")

        if args.save_gif and ep < args.gif_episodes and len(frames) > 0:
            gif_path = out_dir / f"prey_eval_ep{ep+1}.gif"
            imageio.mimsave(gif_path, frames, fps=15)
            print(f"  -> Saved GIF: {gif_path}")

    print("=" * 50)
    print(f"Prey Evaluation Complete!")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Prey Agent")
    parser.add_argument("--N", type=int, default=6, help="Adversary数 (3, 6, 15)")
    parser.add_argument("--episodes", type=int, default=10, help="評価するトータルエピソード数")
    parser.add_argument("--episode_length", type=int, default=50, help="1エピソードの長さ")
    parser.add_argument("--save_gif", action="store_true", help="GIFを保存するかどうか")
    parser.add_argument("--gif_episodes", type=int, default=1, help="GIFとして保存するエピソード数")
    parser.add_argument("--ckpt", type=str, default=None, help="Preyのチェックポイントパス")
    args = parser.parse_args()
    
    evaluate_prey(args)

#python eval_prey.py --N 3 --episodes 10 --save_gif --gif_episodes 3