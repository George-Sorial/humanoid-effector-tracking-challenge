import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

class TrajectoryTracking3DEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Action space: [-1, 1] normalized bounds for PPO
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        
        # NEW: Obs space expanded to 26 to include the "Lookahead Target" (3 vars)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )

        self.t = 0
        self.max_steps = 230
        self.frame_skip = 5  
        self.dt = 1.0 / 240.0
        
        # ─── Control & Filtering Parameters ───
        self.action_alpha = 0.4     # Action smoothing factor (Lower = smoother, Higher = more reactive)
        self.lookahead_steps = 5    # How far into the future the agent sees
        
        self.HOME_JOINTS = np.array([0.0, 0.4, 0.0, -1.0, 0.0, 0.8, 0.0], dtype=np.float32)
        
        self.prev_raw_action = np.zeros(7, dtype=np.float32)
        self.prev_filtered_action = np.zeros(7, dtype=np.float32)
        self.episode_errors = []
        self.robot = None
        self.ee_index = 6

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0
        self.prev_raw_action = np.zeros(7, dtype=np.float32)
        self.prev_filtered_action = np.zeros(7, dtype=np.float32)
        self.episode_errors = []

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        self.robot = p.loadURDF(
            "kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True
        )
        self.ee_index = 6  

        for i, angle in enumerate(self.HOME_JOINTS):
            p.resetJointState(self.robot, i, angle)

        for _ in range(20):
            p.stepSimulation()

        obs = self._get_obs()
        return obs, {}

    def _get_target(self, t_offset):
        current_time = t_offset * self.dt * self.frame_skip
        radius = 0.15
        omega  = 1.5  
        x = radius * np.sin(omega * current_time)
        y = 0.45                                        
        z = 0.55 + (radius / 2) * np.sin(2 * omega * current_time)
        return np.array([x, y, z], dtype=np.float32)

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot, range(7))
        joint_pos = np.array([s[0] for s in joint_states], dtype=np.float32)
        joint_vel = np.array([s[1] for s in joint_states], dtype=np.float32)

        ee_state = p.getLinkState(self.robot, self.ee_index)
        ee_pos   = np.array(ee_state[0], dtype=np.float32)

        target_pos = self._get_target(self.t)
        pos_error  = target_pos - ee_pos
        
        # NEW: Get the future target position for predictive control
        future_target_pos = self._get_target(self.t + self.lookahead_steps)

        return np.concatenate([
            joint_pos, 
            joint_vel, 
            ee_pos, 
            target_pos, 
            pos_error, 
            future_target_pos  # Added predictive lookahead
        ])

    def step(self, action):
        self.t += 1

        # ─── ENHANCEMENT: Action Smoothing (Low-Pass Filter) ───
        # Blends the current network action with the previous executed action
        filtered_action = (self.action_alpha * action) + ((1.0 - self.action_alpha) * self.prev_filtered_action)
        self.prev_filtered_action = filtered_action.copy()

        # Rescale the smoothed action to physical joint limits
        rescaled_action = filtered_action * 0.05

        noise = np.random.normal(0, 0.005, size=action.shape).astype(np.float32)
        noisy_action = np.clip(rescaled_action + noise, -0.05, 0.05)

        current_joints = np.array(
            [p.getJointState(self.robot, i)[0] for i in range(7)], dtype=np.float32
        )
        target_joints = current_joints + noisy_action

        p.setJointMotorControlArray(
            self.robot, range(7),
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joints
        )
        
        for _ in range(self.frame_skip):
            p.stepSimulation()

        obs        = self._get_obs()
        joint_vel  = obs[7:14]
        ee_pos     = obs[14:17]
        target_pos = obs[17:20]

        dist   = float(np.linalg.norm(ee_pos - target_pos))
        effort = float(np.sum(joint_vel ** 2))

        sigma_0       = 0.05          
        lambd         = 0.01          
        sigma_dynamic = sigma_0 + lambd * effort

        r_precision  = np.exp(-(dist ** 2) / (sigma_dynamic ** 2))
        
        # Penalize the raw action jerk so the network learns to output smooth intent
        action_jerk  = float(np.sum((action - self.prev_raw_action) ** 2))

        reward = (3.0 * r_precision) - (1.0 * dist) - (0.05 * action_jerk) - (0.005 * effort)

        self.prev_raw_action = action.copy()
        self.episode_errors.append(dist)

        terminated = False
        truncated = self.t >= self.max_steps
        
        info = {
            "mean_episode_error": np.mean(self.episode_errors) if truncated else 0.0
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect()