import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

class TrajectoryTracking3DEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.physics_client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

        self.t = 0
        self.max_steps = 230
        self.frame_skip = 5  
        self.dt = 1.0 / 240.0
        
        self.action_alpha = 0.35 #was 0.5
        self.lookahead_steps = 8 #was 5
        self.HOME_JOINTS = np.array([0.0, 0.4, 0.0, -1.0, 0.0, 0.8, 0.0], dtype=np.float32)
        
        self.prev_raw_action = np.zeros(7, dtype=np.float32)
        self.prev_filtered_action = np.zeros(7, dtype=np.float32)
        self.episode_errors = []
        self.prev_dist = 0.0 
        self.robot = None
        self.ee_index = 6

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.prev_raw_action = np.zeros(7, dtype=np.float32)
        self.prev_filtered_action = np.zeros(7, dtype=np.float32)
        self.episode_errors = []
        self.current_radius = np.random.uniform(0.12, 0.18)
        self.current_omega = np.random.uniform(1.2, 1.8)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        
        for i, angle in enumerate(self.HOME_JOINTS):
            p.resetJointState(self.robot, i, angle)
        for _ in range(20): p.stepSimulation()

        obs = self._get_obs()
        self.prev_dist = float(np.linalg.norm(obs[14:17] - obs[17:20]))
        return obs, {}

    def _get_target(self, t_offset):
        time = t_offset * self.dt * self.frame_skip
        x = self.current_radius * np.sin(self.current_omega * time)
        y = 0.45                                        
        z = 0.55 + (self.current_radius / 2) * np.sin(2 * self.current_omega * time)
        return np.array([x, y, z], dtype=np.float32)

    def _get_obs(self):
        js = p.getJointStates(self.robot, range(7))
        j_pos, j_vel = np.array([s[0] for s in js]), np.array([s[1] for s in js])
        ee_pos = np.array(p.getLinkState(self.robot, self.ee_index)[0])
        target = self._get_target(self.t)
        return np.concatenate([j_pos, j_vel, ee_pos, target, target - ee_pos, self._get_target(self.t + self.lookahead_steps)])

    def step(self, action):
        self.t += 1
        filtered = (self.action_alpha * action) + ((1.0 - self.action_alpha) * self.prev_filtered_action)
        self.prev_filtered_action = filtered.copy()
        #was 0.08
        noisy = np.clip(filtered * 0.08 + np.random.normal(0, 0.005, 7), -0.08, 0.08)
        cur_j = np.array([p.getJointState(self.robot, i)[0] for i in range(7)])
        p.setJointMotorControlArray(self.robot, range(7), p.POSITION_CONTROL, targetPositions=cur_j + noisy)
        for _ in range(self.frame_skip): p.stepSimulation()

        obs = self._get_obs()
        dist = float(np.linalg.norm(obs[14:17] - obs[17:20]))
        jitter = abs(dist - self.prev_dist)
        self.prev_dist = dist
        
        effort = float(np.sum(obs[7:14] ** 2))
        r_prec = np.exp(-(dist ** 2) / ((0.02 + 0.01 * effort) ** 2))
        # Amplified precision tracking rewards while preserving the heavy jitter suppression
        reward = (5.0 * r_prec) - (3.0 * dist) - (10.0 * (dist**2)) - (0.05 * np.sum((action - self.prev_raw_action)**2)) - (0.005 * effort) - (15.0 * jitter)
        self.prev_raw_action = action.copy()
        self.episode_errors.append(dist)
        return obs, reward, False, self.t >= self.max_steps, {}

    def close(self): p.disconnect()