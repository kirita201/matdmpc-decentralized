# train.py
import os
import random
from pathlib import Path

import imageio
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from envs.make_env import make_env
from algorithm.buffer import ReplayBuffer
from algorithm.diagnostics import Diagnostics
from algorithm.matdmpc_asynch import AsynchMATDMPC
from algorithm.matdmpc_synch import SynchMATDMPC
from algorithm.maddpg import MADDPG
from algorithm.mappo import MAPPO


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


class MockConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_cfg():
    import yaml

    with open("cfgs/default.yaml", "r", encoding="utf-8") as f:
        return MockConfig(**yaml.safe_load(f))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def recorded_reward(reward, reward_type):
    rewards = np.asarray(reward)

    if reward_type == "global":
        return float(rewards.reshape(-1)[0])

    return float(rewards.sum())


def evaluate(
    env,
    agent,
    num_episodes,
    step,
    log_dir,
    reward_type,
    save_gif=False,
):
    episode_rewards = []
    frames = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        t = 0

        while not done:
            if save_gif and ep == 0:
                frame = env.render()

                if frame is not None:
                    if isinstance(frame, list):
                        frames.extend(frame)
                    else:
                        frames.append(frame)

            action = agent.plan(
                obs,
                positions=info.get("agent_positions"),
                eval_mode=True,
                step=step,
                t0=(t == 0),
            )

            obs, reward, done, info = env.step(
                action.cpu().numpy()
            )
            ep_reward += recorded_reward(
                reward,
                reward_type,
            )
            t += 1

        episode_rewards.append(ep_reward)

    if save_gif and frames:
        gif_path = os.path.join(
            log_dir,
            f"eval_step_{step}.gif",
        )
        imageio.mimsave(
            gif_path,
            frames,
            fps=15,
        )

    return np.mean(episode_rewards)


def train():
    cfg = load_cfg()
    set_seed(cfg.seed)

    task_name = getattr(
        cfg,
        "task",
        "simple_spread",
    )
    obs_type = getattr(
        cfg,
        "obs_type",
        "local",
    )
    reward_type = getattr(
        cfg,
        "reward_type",
        "individual",
    )
    algo_type = getattr(
        cfg,
        "algo_type",
        "asynch",
    )
    comm_type = getattr(
        cfg,
        "comm_type",
        "local",
    )

    exp_name = (
        f"{task_name}_N{cfg.num_agents}"
        f"_obs:{obs_type}"
        f"_rew:{reward_type}"
        f"_algo:{algo_type}"
        f"_comm:{comm_type}"
    )

    log_dir = Path("logs") / exp_name
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ckpt_dir = (
        Path(
            getattr(
                cfg,
                "ckpt_dir",
                "checkpoints",
            )
        )
        / exp_name
    )
    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_ckpt_path = ckpt_dir / "model_latest.pt"
    state_ckpt_path = ckpt_dir / "state_latest.pt"
    buffer_ckpt_path = ckpt_dir / "buffer_latest.pt"

    writer = None
    diagnostics = None

    try:
        temp_env = make_env(cfg)
        cfg.obs_shape = list(temp_env.obs_shape)
        cfg.action_dim = temp_env.action_dim

        if (
            hasattr(temp_env, "env")
            and hasattr(temp_env.env, "close")
        ):
            temp_env.env.close()

        env = make_env(cfg)
        eval_env = make_env(
            cfg,
            render_mode="rgb_array",
        )

        buffer = None

        if algo_type == "asynch":
            agent = AsynchMATDMPC(cfg)
            buffer = ReplayBuffer(cfg)

        elif algo_type == "synch":
            agent = SynchMATDMPC(cfg)
            buffer = ReplayBuffer(cfg)

        elif algo_type == "maddpg":
            agent = MADDPG(cfg)
            buffer = ReplayBuffer(cfg)

        elif algo_type == "mappo":
            agent = MAPPO(cfg)

        else:
            raise ValueError(
                f"Unknown algo_type: {algo_type}"
            )

        # step は従来の学習・評価スケジュール用。
        # experienced_steps は学習環境で実際に経験した
        # env.step() の累計回数で、TensorBoard の横軸に使う。
        start_step = 0
        experienced_steps = 0
        episode_idx = 0
        resumed = False

        # ------------------------------------------------------------
        # チェックポイントからの再開
        # ------------------------------------------------------------
        if getattr(cfg, "resume", False):
            if (
                model_ckpt_path.exists()
                and state_ckpt_path.exists()
            ):
                print(
                    ">>> Resuming training "
                    "from checkpoints..."
                )

                state = torch.load(
                    state_ckpt_path,
                    map_location="cpu",
                )

                if "env_steps" not in state:
                    raise ValueError(
                        "チェックポイントに env_steps がありません。"
                        "旧形式の TensorBoard ログは横軸が異なるため、"
                        "そのまま再開して混在させることはできません。"
                    )

                agent.load(model_ckpt_path)

                # state["step"] は保存済みエピソードの
                # スケジュール上の step。次のエピソードから再開する。
                start_step = (
                    int(state["step"])
                    + int(cfg.episode_length)
                )
                experienced_steps = int(
                    state["env_steps"]
                )
                episode_idx = int(
                    state["episode_idx"]
                )
                resumed = True

                if (
                    buffer is not None
                    and buffer_ckpt_path.exists()
                ):
                    print(
                        ">>> Resuming replay "
                        "buffer state..."
                    )
                    buffer.load(buffer_ckpt_path)

                elif buffer is not None:
                    print(
                        ">>> Warning: Replay buffer "
                        "checkpoint not found. "
                        "Starting with an empty buffer."
                    )

        # 保存済みの experienced_steps までのイベントは残し、
        # それより後に書かれたイベントだけを再開時に隠す。
        writer = SummaryWriter(
            log_dir=str(log_dir),
            purge_step=(
                experienced_steps + 1
                if resumed
                else None
            ),
        )

        if algo_type in ("asynch", "synch"):
            diagnostics = Diagnostics(
                log_dir,
                writer,
                resume_env_step=(
                    experienced_steps
                    if resumed
                    else None
                ),
            )
            agent.diagnostics = diagnostics

        initial_random_ratio = 0.8
        decay_steps = 150000

        # ------------------------------------------------------------
        # メインループ
        # ------------------------------------------------------------
        for step in range(
            start_step,
            cfg.train_steps + cfg.episode_length,
            cfg.episode_length,
        ):
            decay_ratio = max(
                0.0,
                1.0 - (step / decay_steps),
            )
            current_ratio = (
                initial_random_ratio
                * decay_ratio
            )

            env.random_ratio = current_ratio
            eval_env.random_ratio = current_ratio

            obs, info = env.reset()
            done = False
            t = 0
            ep_reward = 0

            ep_goals = []
            ep_cols = []
            ep_pred_catches = 0

            # --------------------------------------------------------
            # １エピソード分の環境実行
            # --------------------------------------------------------
            while not done:
                positions = info.get(
                    "agent_positions",
                    None,
                )

                if algo_type == "mappo":
                    action, log_prob, value = agent.plan(
                        obs,
                        positions=positions,
                        step=step,
                        t0=(t == 0),
                    )
                    act_np = action.cpu().numpy()

                else:
                    action = agent.plan(
                        obs,
                        positions=positions,
                        step=step,
                        t0=(t == 0),
                    )
                    act_np = action.cpu().numpy()

                (
                    next_obs,
                    reward,
                    done,
                    next_info,
                ) = env.step(act_np)

                # 評価環境のステップやモデル更新回数は含めない。
                # 学習環境で経験を１ステップ得るたびに加算する。
                experienced_steps += 1

                if algo_type == "mappo":
                    pos_t = (
                        torch.tensor(
                            positions,
                            dtype=torch.float32,
                            device=agent.device,
                        )
                        if positions is not None
                        else None
                    )

                    agent.buffer.store(
                        torch.tensor(
                            obs,
                            dtype=torch.float32,
                            device=agent.device,
                        ),
                        action,
                        log_prob,
                        torch.tensor(
                            reward,
                            dtype=torch.float32,
                            device=agent.device,
                        ),
                        value,
                        torch.tensor(
                            1.0 - float(done),
                            dtype=torch.float32,
                            device=agent.device,
                        ),
                        pos_t,
                    )

                else:
                    buffer.add(
                        obs,
                        act_np,
                        reward,
                        done,
                        info,
                    )

                obs = next_obs
                info = next_info

                ep_reward += recorded_reward(
                    reward,
                    reward_type,
                )

                if "goals_occupied_now" in info:
                    ep_goals.append(
                        info["goals_occupied_now"]
                    )

                if "collisions_now" in info:
                    ep_cols.append(
                        info["collisions_now"]
                    )

                if "predator_catch" in info:
                    ep_pred_catches += (
                        info["predator_catch"]
                    )

                t += 1

            # このエピソード終了時点までに経験したステップ数。
            episode_tb_step = experienced_steps

            # --------------------------------------------------------
            # エピソード指標
            # --------------------------------------------------------
            writer.add_scalar(
                "Train/EpisodeReward",
                ep_reward,
                episode_tb_step,
            )

            if ep_goals:
                writer.add_scalar(
                    "Train/GoalsOccupied_End",
                    ep_goals[-1],
                    episode_tb_step,
                )
                writer.add_scalar(
                    "Train/Collisions_Sum",
                    np.sum(ep_cols),
                    episode_tb_step,
                )

            if (
                ep_pred_catches > 0
                or "predator" in task_name
            ):
                writer.add_scalar(
                    "Train/PredatorCatches_Sum",
                    ep_pred_catches,
                    episode_tb_step,
                )

            # --------------------------------------------------------
            # モデル更新
            # --------------------------------------------------------
            if algo_type == "mappo":
                rollout_length = getattr(
                    cfg,
                    "rollout_length",
                    cfg.episode_length,
                )

                if (
                    len(agent.buffer.rewards)
                    >= rollout_length
                ):
                    next_obs_t = torch.tensor(
                        obs,
                        dtype=torch.float32,
                        device=agent.device,
                    )

                    next_pos_t = (
                        torch.tensor(
                            info["agent_positions"],
                            dtype=torch.float32,
                            device=agent.device,
                        )
                        if "agent_positions" in info
                        else None
                    )

                    loss_info = agent.update(
                        next_obs_t,
                        next_pos_t,
                        float(done),
                    )

                    for name, value in loss_info.items():
                        writer.add_scalar(
                            f"Loss/{name}",
                            value,
                            episode_tb_step,
                        )

            elif step >= cfg.seed_steps:
                num_updates = (
                    cfg.seed_steps
                    if step == cfg.seed_steps
                    else cfg.episode_length
                )

                update_iterator = (
                    tqdm(
                        range(num_updates),
                        desc="Initial Updates",
                    )
                    if num_updates > cfg.episode_length
                    else range(num_updates)
                )

                loss_accum = {}

                for i in update_iterator:
                    # agent.update() に渡す値は従来どおり。
                    update_step = (
                        i
                        if step == cfg.seed_steps
                        else (
                            step
                            - cfg.episode_length
                            + i
                        )
                    )

                    if diagnostics is not None:
                        diagnostics.set_context(
                            update_index=update_step,
                            env_step=experienced_steps,
                            tensorboard_step=(
                                experienced_steps
                            ),
                        )

                    loss_info = agent.update(
                        buffer,
                        update_step,
                    )

                    for (
                        loss_name,
                        loss_val,
                    ) in loss_info.items():
                        loss_accum[loss_name] = (
                            loss_accum.get(
                                loss_name,
                                0.0,
                            )
                            + loss_val.item()
                        )

                for (
                    loss_name,
                    loss_sum,
                ) in loss_accum.items():
                    writer.add_scalar(
                        f"Loss/{loss_name}",
                        loss_sum / num_updates,
                        episode_tb_step,
                    )

            episode_idx += 1

            if episode_idx % 5 == 0:
                print(
                    f"Env steps: {experienced_steps}, "
                    f"Episode: {episode_idx}, "
                    f"Reward: {ep_reward:.3f}"
                )

            # --------------------------------------------------------
            # チェックポイント
            # --------------------------------------------------------
            if (
                step > 0
                and step
                % getattr(
                    cfg,
                    "save_freq",
                    50000,
                )
                == 0
            ):
                agent.save(model_ckpt_path)

                torch.save(
                    {
                        "step": step,
                        "env_steps": experienced_steps,
                        "episode_idx": episode_idx,
                    },
                    state_ckpt_path,
                )

                if buffer is not None:
                    buffer.save(
                        buffer_ckpt_path
                    )

                print(
                    ">>> Checkpoints "
                    "(Model, State, Buffer) "
                    f"saved at Env steps {experienced_steps}"
                )

            # --------------------------------------------------------
            # 評価
            # --------------------------------------------------------
            if (
                step % cfg.eval_freq == 0
                and step > 0
            ):
                eval_reward = evaluate(
                    eval_env,
                    agent,
                    cfg.eval_episodes,
                    step,
                    log_dir,
                    reward_type,
                    save_gif=True,
                )
                print(
                    f">>> EVAL at Env steps "
                    f"{experienced_steps}: "
                    f"Reward = {eval_reward:.3f}"
                )

    finally:
        if diagnostics is not None:
            diagnostics.close()

        if writer is not None:
            writer.close()


if __name__ == "__main__":
    train()