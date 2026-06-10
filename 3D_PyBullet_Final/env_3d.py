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
        
        # ─── FIX 1: Normalize Action Space to [-1, 1] ───
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32
        )

        self.t = 0
        self.max_steps = 230
        self.frame_skip = 5  # ─── FIX 2: Allow time for the robot to move ───
        self.dt = 1.0 / 240.0
        
        # A neutral pose that puts the Kuka EE roughly in the workspace
        # where the trajectory lives — avoids cold-start exploration waste
        self.HOME_JOINTS = np.array([0.0, 0.4, 0.0, -1.0, 0.0, 0.8, 0.0], dtype=np.float32)
        
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.episode_errors = []
        self.robot = None
        self.ee_index = 6

    def reset(self, seed=None, options=None):
        # Handle gymnasium seeding rules
        super().reset(seed=seed)

        # 1. Reset trackers
        self.t = 0
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.episode_errors = []

        # 2. Reset the PyBullet simulator and reload assets
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        # Load the 7-DOF Kuka iiwa Arm
        self.robot = p.loadURDF(
            "kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True
        )
        self.ee_index = 6  # End-effector link index

        # 3. Teleport robot to home position so it starts near the circle path
        for i, angle in enumerate(self.HOME_JOINTS):
            p.resetJointState(self.robot, i, angle)

        # 4. Settle physics simulation momentarily
        for _ in range(20):
            p.stepSimulation()

        # 5. Return initial observation and info dict
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_target(self, t):
        # ─── FIX 3: Define trajectory using actual physical seconds ───
        current_time = t * self.dt * self.frame_skip
        
        radius = 0.15
        omega  = 1.5  # Realistic physical frequency (radians per second)
        
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

        return np.concatenate([joint_pos, joint_vel, ee_pos, target_pos, pos_error])

    def step(self, action):
        self.t += 1

        # ─── FIX 1 Continued: Rescale the agent's action internally ───
        rescaled_action = action * 0.05

        # Apply action noise safely within scaled limits
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
        
        # ─── FIX 2 Continued: Run multiple physics steps per gym step ───
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
        action_jerk  = float(np.sum((action - self.prev_action) ** 2))

        # ─── FIX 4: Add dense tracking guidance (-1.0 * dist) ───
        reward = (3.0 * r_precision) - (1.0 * dist) - (0.05 * action_jerk) - (0.005 * effort)

        self.prev_action = action.copy()
        self.episode_errors.append(dist)

        terminated = False
        truncated = self.t >= self.max_steps
        
        info = {
            "mean_episode_error": np.mean(self.episode_errors) if truncated else 0.0
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect()