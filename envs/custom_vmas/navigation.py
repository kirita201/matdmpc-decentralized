# envs/custom_vmas/navigation.py
import torch
import math
from vmas.simulator.core import Agent, Landmark, Sphere, Box, World
from vmas.simulator.scenario import BaseScenario

def compute_lidar(agent_pos, target_positions, target_radii, num_rays, max_range, walls=None):
    """シンプルな2D TensorベースのRaycast Lidarシミュレーション（壁の検出にも対応）"""
    B = agent_pos.shape[0]
    device = agent_pos.device
    
    angles = torch.linspace(0, 2 * math.pi, num_rays + 1, device=device)[:-1]
    rays = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) # [num_rays, 2]
    min_dists = torch.full((B, num_rays), max_range, device=device)
    
    # 1. 円形オブジェクト（エージェントや障害物）の検出
    for t_pos, t_rad in zip(target_positions, target_radii):
        V = t_pos.unsqueeze(1) - agent_pos.unsqueeze(1) # [B, 1, 2]
        proj = torch.sum(V * rays.unsqueeze(0), dim=-1) # [B, num_rays]
        dist_to_ray_sq = torch.sum(V**2, dim=-1) - proj**2 # [B, num_rays]
        
        hit_mask = (dist_to_ray_sq <= t_rad**2) & (proj > 0)
        dist_to_surf = proj - torch.sqrt(torch.clamp(t_rad**2 - dist_to_ray_sq, min=0.0))
        valid_hit = hit_mask & (dist_to_surf > 0) & (dist_to_surf < min_dists)
        min_dists = torch.where(valid_hit, dist_to_surf, min_dists)
        
    # 2. 直線壁の検出 (wallsが指定されている場合、[-2.0, 2.0] の外周壁との交点を計算)
    if walls is not None:
        # 各レイの方向ベクトル [B, num_rays, 2]
        expanded_rays = rays.unsqueeze(0).expand(B, -1, -1)
        
        # X軸方向の壁（左右の壁: x = -2.0, x = 2.0）との交点距離
        # ray_x * t + agent_x = wall_x  =>  t = (wall_x - agent_x) / ray_x
        for wall_x in [-2.0, 2.0]:
            t_x = (wall_x - agent_pos[:, 0:1]) / (expanded_rays[:, :, 0] + 1e-8)
            # エージェントの前方かつ有効範囲内の交点をチェック
            valid_x = (t_x > 0) & (t_x < min_dists)
            # 交点におけるY座標が壁の範囲 [-2.0, 2.0] 内にあるか確認
            intersect_y = agent_pos[:, 1:2] + t_x * expanded_rays[:, :, 1]
            valid_x &= (intersect_y >= -2.0) & (intersect_y <= 2.0)
            min_dists = torch.where(valid_x, t_x, min_dists)
            
        # Y軸方向の壁（上下の壁: y = -2.0, y = 2.0）との交点距離
        for wall_y in [-2.0, 2.0]:
            t_y = (wall_y - agent_pos[:, 1:2]) / (expanded_rays[:, :, 1] + 1e-8)
            valid_y = (t_y > 0) & (t_y < min_dists)
            intersect_x = agent_pos[:, 0:2] + t_y * expanded_rays[:, :, 0]
            valid_y &= (intersect_x >= -2.0) & (intersect_x <= 2.0)
            min_dists = torch.where(valid_y, t_y, min_dists)
            
    return min_dists / max_range

class NavigationScenario(BaseScenario):
    def __init__(self, num_agents, reward_type="individual", obs_type="local"):
        super().__init__()
        self.num_agents = num_agents
        self.reward_type = reward_type
        self.obs_type = obs_type
        
        self.agent_radius = 0.2
        self.obstacle_radius = 0.6
        self.goal_radius = 0.2
        self.max_obstacles = 4
        self.obs_range = 1.0
        self.n_rays_agent = 12
        self.n_rays_obstacle = 12
        
    def make_world(self, batch_dim, device, **kwargs):
        world = World(batch_dim=batch_dim, device=device)
        
        # 周囲の固定壁を配置 [-2.0, 2.0] の外周
        self.walls = []
        wall_color = (0.3, 0.3, 0.3)
        self.walls.append(Landmark(name="wall_top", collide=True, movable=False, shape=Box(length=4.4, width=0.2), color=wall_color))
        self.walls.append(Landmark(name="wall_bottom", collide=True, movable=False, shape=Box(length=4.4, width=0.2), color=wall_color))
        self.walls.append(Landmark(name="wall_left", collide=True, movable=False, shape=Box(length=0.2, width=4.0), color=wall_color))
        self.walls.append(Landmark(name="wall_right", collide=True, movable=False, shape=Box(length=0.2, width=4.0), color=wall_color))
        for wall in self.walls:
            world.add_landmark(wall)

        self.obstacles = []
        for i in range(self.max_obstacles):
            obs = Landmark(name=f"obstacle_{i}", collide=True, movable=False, shape=Sphere(self.obstacle_radius), color=(0.5, 0.5, 0.5))
            self.obstacles.append(obs)
            world.add_landmark(obs)
            
        self.agents_list = []
        for i in range(self.num_agents):
            agent = Agent(name=f"agent_{i}", collide=True, shape=Sphere(self.agent_radius), color=(0.2, 0.2, 0.8), render_action=True)
            self.agents_list.append(agent)
            world.add_agent(agent)
            
        self.goals = []
        for i in range(self.num_agents):
            goal = Landmark(name=f"goal_{i}", collide=False, movable=False, shape=Sphere(self.goal_radius), color=(0.2, 0.8, 0.2))
            self.goals.append(goal)
            world.add_landmark(goal)
        return world
        
    def reset_world_at(self, env_index):
        B = len(env_index) if env_index is not None else self.world.batch_dim
        idx = env_index if env_index is not None else slice(None)
        device = self.world.device
        
        # 壁の位置を確定
        self.walls[0].state.pos[idx] = torch.tensor([0.0, 2.1], device=device).expand(B, 2)
        self.walls[1].state.pos[idx] = torch.tensor([0.0, -2.1], device=device).expand(B, 2)
        self.walls[2].state.pos[idx] = torch.tensor([-2.1, 0.0], device=device).expand(B, 2)
        self.walls[3].state.pos[idx] = torch.tensor([2.1, 0.0], device=device).expand(B, 2)

        placed_entities = []
        
        # 1. 障害物のランダム数配置 (0〜4個)
        num_obs_per_env = torch.randint(0, self.max_obstacles + 1, (B,), device=device)
        
        for i, obs in enumerate(self.obstacles):
            active_mask = (i < num_obs_per_env)
            success = ~active_mask
            pos = torch.full((B, 2), 100.0, device=device) # 非アクティブは場外へ
            
            for _ in range(50):
                if success.all(): break
                mask = ~success
                
                # 壁と障害物の間にエージェントが通れる余白を確保するため、サンプル範囲を [-0.9, 0.9] に制限
                new_pos = torch.rand((B, 2), device=device) * 1.8 - 0.9
                pos[mask] = new_pos[mask]
                
                collision = torch.zeros(B, dtype=torch.bool, device=device)
                for p_pos, p_rad in placed_entities:
                    # 障害物同士の間隔制限 (相手半径 + 自分半径 + エージェント直径0.4 + α)
                    min_dist = self.obstacle_radius + p_rad + (self.agent_radius * 2) + 0.1
                    collision |= (torch.norm(pos - p_pos, dim=-1) < min_dist)
                success = active_mask & (~collision) | (~active_mask)
                
            obs.state.pos[idx] = pos
            placed_entities.append((pos.clone(), self.obstacle_radius))
            obs.state.vel[idx] = 0.0
            
        # 2. エージェントの配置 (壁内 [-1.8, 1.8] でサンプル)
        for agent in self.agents_list:
            success = torch.zeros(B, dtype=torch.bool, device=device)
            pos = torch.zeros((B, 2), device=device)
            for _ in range(50):
                if success.all(): break
                mask = ~success
                new_pos = torch.rand((B, 2), device=device) * 3.6 - 1.8
                pos[mask] = new_pos[mask]
                
                collision = torch.zeros(B, dtype=torch.bool, device=device)
                for p_pos, p_rad in placed_entities:
                    collision |= (torch.norm(pos - p_pos, dim=-1) < (self.agent_radius + p_rad + 0.1))
                success = ~collision
            agent.state.pos[idx] = pos
            placed_entities.append((pos.clone(), self.agent_radius))
            agent.state.vel[idx] = 0.0

        # 3. ゴールの配置
        for i, goal in enumerate(self.goals):
            success = torch.zeros(B, dtype=torch.bool, device=device)
            pos = torch.zeros((B, 2), device=device)
            for _ in range(50):
                if success.all(): break
                mask = ~success
                new_pos = torch.rand((B, 2), device=device) * 3.6 - 1.8
                pos[mask] = new_pos[mask]
                
                collision = torch.zeros(B, dtype=torch.bool, device=device)
                # 障害物および他のゴールとの衝突判定
                for p_pos, p_rad in placed_entities[:self.max_obstacles]:
                    collision |= (torch.norm(pos - p_pos, dim=-1) < (self.goal_radius + p_rad + 0.1))
                for g in self.goals[:i]:
                    collision |= (torch.norm(pos - g.state.pos[idx], dim=-1) < (self.goal_radius * 2 + 0.1))
                success = ~collision
            goal.state.pos[idx] = pos
            placed_entities.append((pos.clone(), self.goal_radius))
            goal.state.vel[idx] = 0.0
            
        if not hasattr(self, "prev_dists"):
            self.prev_dists = [torch.zeros(self.world.batch_dim, device=device) for _ in range(self.num_agents)]
        
        for i, agent in enumerate(self.agents_list):
            self.prev_dists[i][idx] = torch.norm(agent.state.pos[idx] - self.goals[i].state.pos[idx], dim=-1)

        if not hasattr(self, "occupied_goals"):
            self.occupied_goals = torch.zeros((self.world.batch_dim, self.num_agents), device=device, dtype=torch.bool)
            self.collisions = torch.zeros((self.world.batch_dim, self.num_agents), device=device, dtype=torch.int32)

        self.occupied_goals[idx] = False
        self.collisions[idx] = 0

    def reward(self, agent):
        i = self.agents_list.index(agent)
        goal = self.goals[i]
        
        curr_dist = torch.norm(agent.state.pos - goal.state.pos, dim=-1)
        reward_dist = self.prev_dists[i] - curr_dist
        self.prev_dists[i] = curr_dist
        
        occupied = curr_dist < self.goal_radius
        self.occupied_goals[:, i] = occupied
        reward_occ = occupied.float() * 0.15
        
        collision_penalty = torch.zeros(self.world.batch_dim, device=self.world.device)
        num_cols = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.int32)
        
        # エージェント間衝突
        for other in self.agents_list:
            if other is not agent:
                col = torch.norm(agent.state.pos - other.state.pos, dim=-1) < (self.agent_radius + self.agent_radius)
                collision_penalty[col] -= 0.05
                num_cols[col] += 1
                
        # 障害物との衝突
        for obs in self.obstacles:
            col = torch.norm(agent.state.pos - obs.state.pos, dim=-1) < (self.agent_radius + self.obstacle_radius)
            collision_penalty[col] -= 0.05
            num_cols[col] += 1
            
        # ▼ 追加: 周囲の壁との衝突ペナルティ (外周境界2.0からエージェント半径0.2以内)
        wall_col = (torch.abs(agent.state.pos[:, 0]) > (2.0 - self.agent_radius)) | \
                   (torch.abs(agent.state.pos[:, 1]) > (2.0 - self.agent_radius))
        collision_penalty[wall_col] -= 0.05
        num_cols[wall_col] += 1
            
        self.collisions[:, i] = num_cols
        return reward_dist + reward_occ + collision_penalty

    def observation(self, agent):
        i = self.agents_list.index(agent)
        vel, pos = agent.state.vel, agent.state.pos
        goal_rel_pos = self.goals[i].state.pos - pos
        
        if self.obs_type == "global":
            obs = [vel, pos, goal_rel_pos]
            for other in self.agents_list:
                if other is not agent: obs.append(other.state.pos - pos)
            for obs_lm in self.obstacles: obs.append(obs_lm.state.pos - pos)
            return torch.cat(obs, dim=-1)
            
        other_pos = [a.state.pos for a in self.agents_list if a is not agent]
        # 他エージェントのLidar検出
        agent_lidar = compute_lidar(pos, other_pos, [self.agent_radius]*len(other_pos), self.n_rays_agent, self.obs_range)
        
        # ▼ 修正: 障害物だけでなく、外周壁もLidarの検出対象（引数walls）として追加
        obs_pos = [o.state.pos for o in self.obstacles]
        obstacle_and_wall_lidar = compute_lidar(
            pos, obs_pos, [self.obstacle_radius]*len(obs_pos), 
            self.n_rays_obstacle, self.obs_range, walls=self.walls
        )
        
        return torch.cat([vel, pos, goal_rel_pos, agent_lidar, obstacle_and_wall_lidar], dim=-1)

    def get_metrics(self):
        return {
            'goals_occupied_now': self.occupied_goals.sum(dim=-1).float(),
            'collisions_now': self.collisions.sum(dim=-1).float()
        }