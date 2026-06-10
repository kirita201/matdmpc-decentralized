# envs/mpe_wrapper.py
"""
MA-TDMPC用 MPEラッパー

論文 Table I の実験設定を完全再現:
  ┌──────────────────────────┬───────┬────────┬────────┬──────────────────┬─────────┐
  │ Task                     │ N_agt │ N_lm   │ N_prey │ Visible(agt/lm/  │ obs_dim │
  │                          │       │        │        │ prey)            │         │
  ├──────────────────────────┼───────┼────────┼────────┼──────────────────┼─────────┤
  │ Cooperative Navigation   │   3   │   3    │   —    │  (2, 3, —)       │   14    │
  │                          │   6   │   6    │   —    │  (5, 6, —)       │   26    │
  │                          │  15   │  15    │   —    │  (5, 6, —)       │   26    │
  ├──────────────────────────┼───────┼────────┼────────┼──────────────────┼─────────┤
  │ Predator Prey            │   3   │   2    │   1    │  (2, 2, 1)       │   16    │
  │ (adversaries only)       │   6   │   3    │   2    │  (5, 2, 2)       │   28    │
  │                          │  15   │   5    │   5    │  (6, 3, 3)       │   34    │
  └──────────────────────────┴───────┴────────┴────────┴──────────────────┴─────────┘

観測範囲制限:
  - 観測範囲内のエンティティのみを観測に含める
  - 範囲外のエンティティはゼロパディング
  - 観測ベクトルのサイズは (n_visible_agents * 2 + n_visible_lm * 2 + ...) で固定
"""

import numpy as np
from pathlib import Path
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from gymnasium.utils import EzPickle


# ─────────────────────────────────────────────────────────
# 観測範囲制限付き Cooperative Navigation シナリオ
# ─────────────────────────────────────────────────────────
class SpreadScenario(BaseScenario):
    """
    論文の Cooperative Navigation シナリオ。
    各エージェントは observation_range 内の
    n_visible_agents 体・n_visible_landmarks 個のランドマークのみ観測する。
    obs = self_vel(2) + self_pos(2)
          + lm_rel(n_visible_landmarks * 2)
          + other_rel(n_visible_agents * 2)
    """

    def __init__(self, N, obs_range, n_visible_agents, n_visible_landmarks):
        self.N = N
        self.obs_range = obs_range
        self.n_vis_agents = n_visible_agents
        self.n_vis_lm = n_visible_landmarks

    def make_world(self):
        world = World()
        world.dim_c = 2
        world.collaborative = True
        world.agents = [Agent() for _ in range(self.N)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        world.landmarks = [Landmark() for _ in range(self.N)]
        for i, lm in enumerate(world.landmarks):
            lm.name = f"landmark_{i}"
            lm.collide = False
            lm.movable = False
        return world

    def reset_world(self, world, np_random):
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.state.p_pos = np_random.uniform(-1, 1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for lm in world.landmarks:
            lm.color = np.array([0.25, 0.25, 0.25])
            lm.state.p_pos = np_random.uniform(-1, 1, world.dim_p)
            lm.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, a1, a2):
        d = np.linalg.norm(a1.state.p_pos - a2.state.p_pos)
        return d < a1.size + a2.size

    def reward(self, agent, world):
        rew = 0.0
        if agent.collide:
            for a in world.agents:
                if a is not agent and self.is_collision(a, agent):
                    rew -= 1.0
        return rew

    def global_reward(self, world):
        rew = 0.0
        for lm in world.landmarks:
            dists = [np.linalg.norm(a.state.p_pos - lm.state.p_pos) for a in world.agents]
            min_dist = min(dists)
            rew -= min_dist

            # 2. 占有ボーナス（ここを追加！）
            # 距離が一定以下（例: 0.15）なら「カバーした」とみなして加点
            if min_dist < 0.15: 
                rew += 1.0
        return rew

    def observation(self, agent, world):
        # ── ランドマーク: 距離が近い順に n_vis_lm 個 ──
        lm_rel = []
        for lm in world.landmarks:
            lm_rel.append((np.linalg.norm(lm.state.p_pos - agent.state.p_pos),
                           lm.state.p_pos - agent.state.p_pos))
        lm_rel.sort(key=lambda x: x[0])
        lm_obs = []
        for i in range(self.n_vis_lm):
            if i < len(lm_rel) and lm_rel[i][0] <= self.obs_range:
                lm_obs.append(lm_rel[i][1])
            else:
                lm_obs.append(np.zeros(world.dim_p))  # 範囲外はゼロ

        # ── 他エージェント: 距離が近い順に n_vis_agents 体 ──
        other_rel = []
        for other in world.agents:
            if other is agent:
                continue
            d = np.linalg.norm(other.state.p_pos - agent.state.p_pos)
            other_rel.append((d, other.state.p_pos - agent.state.p_pos))
        other_rel.sort(key=lambda x: x[0])
        other_obs = []
        for i in range(self.n_vis_agents):
            if i < len(other_rel) and other_rel[i][0] <= self.obs_range:
                other_obs.append(other_rel[i][1])
            else:
                other_obs.append(np.zeros(world.dim_p))  # 範囲外はゼロ

        return np.concatenate(
            [agent.state.p_vel, agent.state.p_pos]
            + lm_obs
            + other_obs
        )


# ─────────────────────────────────────────────────────────
# 観測範囲制限付き Predator Prey シナリオ
# ─────────────────────────────────────────────────────────
class PredatorPreyScenario(BaseScenario):
    """
    論文の Predator Prey シナリオ。
    Adversary (捕食者) のみを学習対象エージェントとして扱う。
    obs = self_vel(2) + self_pos(2)
          + lm_rel(n_visible_lm * 2)
          + other_adv_rel(n_visible_adv * 2)
          + prey_rel(n_visible_prey * 2)
          + prey_vel(n_visible_prey * 2)
    """

    def __init__(self, num_adversaries, num_good, num_obstacles,
                 obs_range, n_visible_adv, n_visible_lm, n_visible_prey):
        self.num_adversaries = num_adversaries
        self.num_good = num_good
        self.num_obstacles = num_obstacles
        self.obs_range = obs_range
        self.n_vis_adv = n_visible_adv
        self.n_vis_lm = n_visible_lm
        self.n_vis_prey = n_visible_prey

    def make_world(self):
        world = World()
        world.dim_c = 2
        total_agents = self.num_adversaries + self.num_good
        world.agents = [Agent() for _ in range(total_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = (i < self.num_adversaries)
            base = "adversary" if agent.adversary else "agent"
            idx = i if agent.adversary else i - self.num_adversaries
            agent.name = f"{base}_{idx}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        world.landmarks = [Landmark() for _ in range(self.num_obstacles)]
        for i, lm in enumerate(world.landmarks):
            lm.name = f"landmark_{i}"
            lm.collide = True
            lm.movable = False
            lm.size = 0.2
            lm.boundary = False
        return world

    def reset_world(self, world, np_random):
        for agent in world.agents:
            agent.color = (np.array([0.35, 0.85, 0.35])
                           if not agent.adversary
                           else np.array([0.85, 0.35, 0.35]))
            agent.state.p_pos = np_random.uniform(-1, 1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for lm in world.landmarks:
            lm.color = np.array([0.25, 0.25, 0.25])
            lm.state.p_pos = np_random.uniform(-0.9, 0.9, world.dim_p)
            lm.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, a1, a2):
        d = np.linalg.norm(a1.state.p_pos - a2.state.p_pos)
        return d < a1.size + a2.size

    def good_agents(self, world):
        return [a for a in world.agents if not a.adversary]

    def adversaries(self, world):
        return [a for a in world.agents if a.adversary]

    def reward(self, agent, world):
        if agent.adversary:
            return self._adversary_reward(agent, world)
        else:
            return self._agent_reward(agent, world)

    def _adversary_reward(self, agent, world):
        rew = 0.0
        # チームの「誰か」が捕まえたら全員に加点する
        for adv in self.adversaries(world):
            for prey in self.good_agents(world):
                if self.is_collision(adv, prey):
                    rew += 10.0
        return rew

    def _agent_reward(self, agent, world):
        rew = 0.0

        # 捕食者リスト取得
        advs = self.adversaries(world)
        dists = [np.linalg.norm(a.state.p_pos - agent.state.p_pos) for a in advs]
        min_dist = min(dists)
        
        # 1. 距離報酬: 捕食者と離れるとプラス (上限 1.0)
        # 距離が 1.0 以上離れていれば報酬が頭打ちになるように設定
        rew += min(min_dist, 1.0)

        for adv in self.adversaries(world):
            if self.is_collision(adv, agent):
                rew -= 5.0
        
        # 2. 範囲外ペナルティ (標準仕様の復元)
        # 座標の絶対値が 0.9 を超えるとペナルティが発生し、1.0 を超えると指数関数的に増大する
        def bound(x):
            if x < 0.9:
                return 0.0
            if x < 1.0:
                return (x - 0.9) * 10.0
            return min(np.exp(2 * x - 2), 10.0)
            
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
        return rew

    def observation(self, agent, world):
        if not agent.adversary:
            # ─────────────────────────────────────────────────────────
            # 獲物 (Prey) 側の完全観測ロジック
            # ・観測距離制限なし
            # ・観測エンティティ数の制限なし（すべて観測）
            # ・自分自身を他エージェントの観測対象から除外
            # ─────────────────────────────────────────────────────────
            
            # ── 障害物 (すべて) ──
            lm_obs = sorted(
                [lm.state.p_pos - agent.state.p_pos
                 for lm in world.landmarks if not lm.boundary],
                key=lambda x: np.linalg.norm(x)
            )

            # ── 捕食者 (すべて) ──
            adv_obs = sorted(
                [other.state.p_pos - agent.state.p_pos
                 for other in self.adversaries(world)],
                key=lambda x: np.linalg.norm(x)
            )

            # ── 他の獲物 (自分以外すべて) ──
            # 自分自身 (p is not agent) を除外し、位置と速度を取得
            prey_list = sorted(
                [(p.state.p_pos - agent.state.p_pos, p.state.p_vel)
                 for p in self.good_agents(world) if p is not agent],
                key=lambda x: np.linalg.norm(x[0])
            )
            prey_pos_obs = [p[0] for p in prey_list]
            prey_vel_obs = [p[1] for p in prey_list]

            # 制限やゼロパディングを行わず、すべての情報を結合して返す
            return np.concatenate(
                [agent.state.p_vel, agent.state.p_pos]
                + lm_obs
                + adv_obs
                + prey_pos_obs
                + prey_vel_obs
            )

        else:
            # ─────────────────────────────────────────────────────────
            # 捕食者 (Adversary) 側の部分観測ロジック
            # ─────────────────────────────────────────────────────────
            
            # 【追加】範囲外パディング用のダミー値
            # 位置は「遠く離れた場所」、速度は「停止状態」とする
            dummy_pos = np.full(world.dim_p, 10.0)
            dummy_vel = np.zeros(world.dim_p)
            
            # ── 障害物: 近い順に n_vis_lm 個 ──
            lm_rel = sorted(
                [(np.linalg.norm(lm.state.p_pos - agent.state.p_pos),
                  lm.state.p_pos - agent.state.p_pos)
                 for lm in world.landmarks if not lm.boundary],
                key=lambda x: x[0]
            )
            lm_obs = []
            for i in range(self.n_vis_lm):
                if i < len(lm_rel) and lm_rel[i][0] <= self.obs_range:
                    lm_obs.append(lm_rel[i][1])
                else:
                    # 【修正】ゼロではなくダミー座標でパディング
                    lm_obs.append(dummy_pos)

            # ── 他の捕食者 ──
            other_adv = sorted(
                [(np.linalg.norm(other.state.p_pos - agent.state.p_pos),
                  other.state.p_pos - agent.state.p_pos)
                 for other in self.adversaries(world) if other is not agent],
                key=lambda x: x[0]
            )
            adv_obs = []
            for i in range(self.n_vis_adv):
                if i < len(other_adv) and other_adv[i][0] <= self.obs_range:
                    adv_obs.append(other_adv[i][1])
                else:
                    # 【修正】ゼロではなくダミー座標でパディング
                    adv_obs.append(dummy_pos)

            # ── 獲物: 位置 + 速度 ──
            prey_list = sorted(
                [(np.linalg.norm(p.state.p_pos - agent.state.p_pos),
                  p.state.p_pos - agent.state.p_pos,
                  p.state.p_vel)
                 for p in self.good_agents(world)],
                key=lambda x: x[0]
            )
            prey_pos_obs = []
            prey_vel_obs = []
            for i in range(self.n_vis_prey):
                if i < len(prey_list) and prey_list[i][0] <= self.obs_range:
                    prey_pos_obs.append(prey_list[i][1])
                    prey_vel_obs.append(prey_list[i][2])
                else:
                    # 【修正】位置はダミー座標、速度はゼロでパディング
                    prey_pos_obs.append(dummy_pos)
                    prey_vel_obs.append(dummy_vel)

            return np.concatenate(
                [agent.state.p_vel, agent.state.p_pos]
                + lm_obs
                + adv_obs
                + prey_pos_obs
                + prey_vel_obs
            )


# ─────────────────────────────────────────────────────────
# カスタム PettingZoo 環境クラス生成ヘルパー
# ─────────────────────────────────────────────────────────
def _make_spread_raw_env(N, obs_range, n_visible_agents, n_visible_landmarks,
                         local_ratio, max_cycles, continuous_actions, render_mode=None):
    """SpreadScenario を使った raw_env を動的生成する。"""

    class SpreadRawEnv(SimpleEnv, EzPickle):
        def __init__(self):
            EzPickle.__init__(self)
            scenario = SpreadScenario(N, obs_range, n_visible_agents, n_visible_landmarks)
            world = scenario.make_world()
            SimpleEnv.__init__(
                self,
                scenario=scenario,
                world=world,
                render_mode=render_mode,
                max_cycles=max_cycles,
                continuous_actions=continuous_actions,
                local_ratio=local_ratio,
            )
            self.metadata["name"] = f"spread_N{N}_range{obs_range}"

    return SpreadRawEnv


def _make_predator_prey_raw_env(num_adversaries, num_good, num_obstacles,
                                obs_range, n_visible_adv, n_visible_lm,
                                n_visible_prey, max_cycles, continuous_actions, render_mode=None):
    """PredatorPreyScenario を使った raw_env を動的生成する。"""

    class PredatorPreyRawEnv(SimpleEnv, EzPickle):
        def __init__(self):
            EzPickle.__init__(self)
            scenario = PredatorPreyScenario(
                num_adversaries, num_good, num_obstacles,
                obs_range, n_visible_adv, n_visible_lm, n_visible_prey
            )
            world = scenario.make_world()
            SimpleEnv.__init__(
                self,
                scenario=scenario,
                world=world,
                render_mode=render_mode,
                max_cycles=max_cycles,
                continuous_actions=continuous_actions,
            )
            self.metadata["name"] = (
                f"predprey_N{num_adversaries}_range{obs_range}"
            )

    return PredatorPreyRawEnv


# ─────────────────────────────────────────────────────────
# 論文の実験設定テーブル
# ─────────────────────────────────────────────────────────

# Cooperative Navigation 設定
# {N: (obs_range, n_vis_agents, n_vis_lm)}
SPREAD_CONFIGS = {
    3:  (1.0, 2, 3),   # obs_dim=14
    6:  (1.0, 5, 6),   # obs_dim=26
    15: (1.0, 5, 6),   # obs_dim=26
}

# Predator Prey 設定
# {N_adversaries: (num_good, num_obstacles, obs_range, n_vis_adv, n_vis_lm, n_vis_prey)}
PREDPREY_CONFIGS = {
    3:  (1, 2, 1.0, 2, 2, 1),   # obs_dim=16
    6:  (2, 3, 1.0, 5, 2, 2),   # obs_dim=28
    15: (5, 5, 1.0, 6, 3, 3),   # obs_dim=34
}


# ─────────────────────────────────────────────────────────
# Prey ポリシーロードヘルパー
# ─────────────────────────────────────────────────────────

def _load_prey_policy(ckpt_path, n_prey, device, noise_std, env, prey_agent_ids):
    """
    チェックポイントから PreyPolicy をロードする。
    チェックポイントが存在しない場合はランダムポリシー (未学習 MLP + ノイズ) を返す。

    Parameters
    ----------
    ckpt_path      : Path  チェックポイントファイルパス
    n_prey         : int   good agent 数
    device         : str   "cuda" or "cpu"
    noise_std      : float 推論時ノイズ
    env            : PettingZoo AEC env  obs_dim / action_dim 取得用
    prey_agent_ids : list[str]
    """
    # 遅延インポート (循環 import 回避)
    from envs.prey_policy import PreyPolicy, PreyNet

    env.reset()
    obs_dim    = env.observe(prey_agent_ids[0]).shape[0]
    action_dim = env.action_space(prey_agent_ids[0]).shape[0]

    if Path(ckpt_path).exists():
        policy = PreyPolicy.load(ckpt_path, device=device, noise_std=noise_std)
    else:
        print(
            f"[MPEWrapper] WARNING: prey checkpoint not found at '{ckpt_path}'. "
            f"Falling back to RANDOM prey policy. "
            f"Run `python train_prey.py --N {n_prey}` to train a prey policy."
        )
        policy = PreyPolicy.random(n_prey, obs_dim, action_dim, device=device)
    return policy


# ─────────────────────────────────────────────────────────
# MA-TDMPC 向けラッパー
# ─────────────────────────────────────────────────────────
class MPEWrapper:
    """
    MA-TDMPC 用 MPE ラッパー。

    Parameters
    ----------
    cfg : object
        以下の属性を持つ設定オブジェクト:
        - task          : "simple_spread" | "predator_prey"
        - num_agents    : エージェント数 (3 / 6 / 15)
        - episode_length: エピソード長
        - obs_range     : 観測範囲 (float, 省略可。省略時は設定テーブルの値を使用)

    Attributes
    ----------
    obs_shape : tuple
        観測ベクトルの形状 (obs_dim,)
    action_dim : int
        行動次元数
    N : int
        制御対象エージェント数
    """

    def __init__(self, cfg, render_mode=None):
        self.cfg = cfg
        self.N_cfg = cfg.num_agents
        self.render_mode = render_mode
        task = getattr(cfg, "task", "simple_spread")

        if task == "simple_spread":
            self._init_spread(cfg)
        elif task in ("predator_prey", "simple_tag"):
            self._init_predprey(cfg)
        else:
            raise ValueError(f"Unknown task: {task}. Choose 'simple_spread' or 'predator_prey'.")

        # 観測・行動次元を確認
        self.env.reset()
        sample_obs = self.env.observe(self.env.possible_agents[0])
        self.obs_shape = sample_obs.shape       # e.g., (14,)
        self.action_dim = self.env.action_space(
            self.env.possible_agents[0]
        ).shape[0]                              # e.g., 5
        self.env.reset()

        self.random_ratio = 0.0  # デフォルトは本来のポリシー100%

        print(
            f"[MPEWrapper] task={task}, N={self.N}, "
            f"obs_shape={self.obs_shape}, action_dim={self.action_dim}"
        )

    # ── Cooperative Navigation ──────────────────────────
    def _init_spread(self, cfg):
        N = cfg.num_agents
        assert N in SPREAD_CONFIGS, (
            f"simple_spread supports N ∈ {list(SPREAD_CONFIGS.keys())}, got N={N}"
        )
        obs_range, n_vis_agents, n_vis_lm = SPREAD_CONFIGS[N]
        cfg_range = getattr(cfg, "obs_range", None)
        if cfg_range is not None:
            obs_range = cfg_range   # cfg で上書き可能

        RawEnvClass = _make_spread_raw_env(
            N=N,
            obs_range=obs_range,
            n_visible_agents=n_vis_agents,
            n_visible_landmarks=n_vis_lm,
            local_ratio=getattr(cfg, "local_ratio", 0.5),
            max_cycles=cfg.episode_length,
            continuous_actions=True,
            render_mode=self.render_mode
        )
        env_fn = make_env(RawEnvClass)
        self.env = env_fn()
        self.agents = self.env.possible_agents
        self.N = N
        self._task = "spread"

    # ── Predator Prey ───────────────────────────────────
    def _init_predprey(self, cfg):
        N = cfg.num_agents
        assert N in PREDPREY_CONFIGS, (
            f"predator_prey supports N ∈ {list(PREDPREY_CONFIGS.keys())}, got N={N}"
        )
        num_good, num_obstacles, obs_range, n_vis_adv, n_vis_lm, n_vis_prey = \
            PREDPREY_CONFIGS[N]
        cfg_range = getattr(cfg, "obs_range", None)
        if cfg_range is not None:
            obs_range = cfg_range

        RawEnvClass = _make_predator_prey_raw_env(
            num_adversaries=N,
            num_good=num_good,
            num_obstacles=num_obstacles,
            obs_range=obs_range,
            n_visible_adv=n_vis_adv,
            n_visible_lm=n_vis_lm,
            n_visible_prey=n_vis_prey,
            max_cycles=cfg.episode_length,
            continuous_actions=True,
            render_mode=self.render_mode
        )
        env_fn = make_env(RawEnvClass)
        self.env = env_fn()
        # 捕食者のみを制御対象とする
        self.agents = [a for a in self.env.possible_agents if "adversary" in a]
        self.N = N
        self._task = "predprey"
        self._all_agents = self.env.possible_agents
        self._prey_agents = [a for a in self.env.possible_agents if "adversary" not in a]

        # ── Prey ポリシーのロード ─────────────────────────────
        # cfg.prey_ckpt_path が指定されていればそれを、なければデフォルトパスを探す。
        # チェックポイントが見つからない場合はランダムポリシーにフォールバック。
        device = getattr(cfg, "device", "cpu")
        prey_noise = getattr(cfg, "prey_noise_std", 0.0)

        ckpt_path = getattr(cfg, "prey_ckpt_path", None)
        if ckpt_path is None:
            ckpt_path = Path("checkpoints") / f"prey_N{N}.pt"

        self._prey_policy = _load_prey_policy(
            ckpt_path=Path(ckpt_path),
            n_prey=num_good,
            device=device,
            noise_std=prey_noise,
            env=self.env,
            prey_agent_ids=self._prey_agents,
        )

    # ── 共通インタフェース ──────────────────────────────
    def reset(self):
        """
        Returns
        -------
        obs : np.ndarray, shape (N, obs_dim)
        """
        self.env.reset()
        return self._get_obs()

    def step(self, actions):
        """
        Parameters
        ----------
        actions : np.ndarray, shape (N, action_dim)
            捕食者/全エージェントの行動。

        Returns
        -------
        obs     : np.ndarray, shape (N, obs_dim)
        rewards : np.ndarray, shape (N,)
        done    : bool
        info    : dict
        """
        if self._task == "predprey":
            return self._step_predprey(actions)
        else:
            return self._step_spread(actions)

    # ── spread step ─────────────────────────────────────
    def _step_spread(self, actions):
        for i, agent in enumerate(self.agents):
            self.env.step(actions[i])
        
        rewards = []
        dones = []
        for agent in self.agents:
            rewards.append(self.env.rewards[agent])
            dones.append(
                self.env.terminations[agent] or self.env.truncations[agent]
            )
        obs = self._get_obs()
        return obs, np.array(rewards, dtype=np.float32), bool(np.any(dones)), {}

    # ── predator_prey step ───────────────────────────────
    def _step_predprey(self, actions):
        """捕食者の行動を渡し、獲物は訓練済みポリシー (なければランダム) で動かす。"""
        # 獲物の現在観測を取得してポリシーで行動決定
        prey_obs = np.stack([self.env.observe(a) for a in self._prey_agents])
        prey_actions = self._prey_policy.act(prey_obs)  # [n_prey, action_dim]

        # ──【追加】デフォルト 0.0 の random_ratio を使ってブレンド ──
        random_ratio = getattr(self, "random_ratio", 0.0)
        if random_ratio > 0.0:
            # [-1, 1] の一様乱数（完全ランダム行動）を生成
            random_actions = np.random.uniform(-1.0, 1.0, size=prey_actions.shape)
            # 行動をブレンド
            prey_actions = (1.0 - random_ratio) * prey_actions + random_ratio * random_actions

        adv_idx  = 0
        prey_idx = 0
        rewards  = []
        dones    = []
        for agent in self._all_agents:
            if "adversary" in agent:
                act = actions[adv_idx]
                adv_idx += 1
            else:
                act = prey_actions[prey_idx]
                prey_idx += 1
            self.env.step(act)
        for agent in self.agents:
            rewards.append(self.env.rewards[agent])
            dones.append(
                self.env.terminations[agent] or self.env.truncations[agent]
            )
        obs = self._get_obs()
        return obs, np.array(rewards, dtype=np.float32), bool(np.any(dones)), {}

    # ── 観測収集 ────────────────────────────────────────
    def _get_obs(self):
        return np.stack([self.env.observe(a) for a in self.agents])
    
    def render(self):
        self.env.unwrapped.cam_range = 2.0
        return self.env.render()


# ─────────────────────────────────────────────────────────
# 設定ファクトリ: 論文の全実験タスクに対応する cfg を返す
# ─────────────────────────────────────────────────────────
def make_task_cfg(task: str, N: int, base_cfg):
    """
    task と N を指定して、obs_shape / action_dim / num_agents を
    正しい値に設定した cfg を返す。
    base_cfg の他フィールドはそのまま引き継ぐ。

    Parameters
    ----------
    task : "simple_spread" | "predator_prey"
    N    : 3 | 6 | 15
    base_cfg : object with __dict__
    """
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg.task = task
    cfg.num_agents = N

    # 一時的に env を作って obs_shape / action_dim を取得
    env = MPEWrapper(cfg)
    cfg.obs_shape = list(env.obs_shape)
    cfg.action_dim = env.action_dim
    return cfg


# ─────────────────────────────────────────────────────────
# 動作確認
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    class _Cfg:
        episode_length = 25
        obs_range = None  # テーブルデフォルト使用

    print("=" * 60)
    print("Cooperative Navigation")
    print("=" * 60)
    for N in [3, 6, 15]:
        cfg = _Cfg()
        cfg.task = "simple_spread"
        cfg.num_agents = N
        env = MPEWrapper(cfg)
        obs = env.reset()
        actions = np.random.uniform(-1, 1, (N, env.action_dim))
        next_obs, rewards, done, _ = env.step(actions)
        print(
            f"  N={N:2d} | obs={obs.shape} | "
            f"next_obs={next_obs.shape} | rewards={rewards.shape} | done={done}"
        )

    print()
    print("=" * 60)
    print("Predator Prey")
    print("=" * 60)
    for N in [3, 6, 15]:
        cfg = _Cfg()
        cfg.task = "predator_prey"
        cfg.num_agents = N
        env = MPEWrapper(cfg)
        obs = env.reset()
        actions = np.random.uniform(-1, 1, (N, env.action_dim))
        next_obs, rewards, done, _ = env.step(actions)
        print(
            f"  N={N:2d} | obs={obs.shape} | "
            f"next_obs={next_obs.shape} | rewards={rewards.shape} | done={done}"
        )