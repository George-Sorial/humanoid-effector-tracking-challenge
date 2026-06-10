import gymnasium as gym
import numpy as np

class TrajectoryTrackingEnv(gym.Wrapper):
    def __init__(self):
        env = gym.make("Reacher-v4", render_mode="rgb_array")
        super().__init__(env)
        
        self.t = 0
        self.prev_action = np.zeros(self.action_space.shape)
        self.episode_errors = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.t = 0
        self.prev_action = np.zeros(self.action_space.shape)
        self.episode_errors = []
        
        # Set the initial target position at t=0
        radius = 0.1
        target_x = radius * np.cos(0)
        target_y = radius * np.sin(0)
        
        # Calculate fingertip position using the raw engine return BEFORE overwriting
        fingertip_x = obs[4] - obs[8]
        fingertip_y = obs[5] - obs[9]
        
        # Override the initial observation targets safely
        obs[4] = target_x
        obs[5] = target_y
        obs[8] = target_x - fingertip_x
        obs[9] = target_y - fingertip_y
        
        return obs, info

    def step(self, action):
        self.t += 1
        
        # 1. Generate our true moving trajectory
        radius = 0.1
        omega = 0.15  
        target_x = radius * np.cos(omega * self.t)
        target_y = radius * np.sin(omega * self.t)
        
        # Synchronize MuJoCo's background rendering targets
        try:
            self.env.unwrapped.goal = np.array([target_x, target_y])
            self.env.unwrapped.data.qpos[2] = target_x
            self.env.unwrapped.data.qpos[3] = target_y
        except AttributeError:
            pass
            
        # 2. Add action noise safely (Clip to prevent Gym environment crashes)
        noise = np.random.normal(0, 0.01, size=action.shape)
        noisy_action = np.clip(action + noise, self.action_space.low, self.action_space.high)
        
        # 3. Step the base physics engine
        obs, base_reward, terminated, truncated, info = self.env.step(noisy_action)
        
        # 4. OVERWRITE TRACKING OBSERVATIONS (Give the agent clear sight)
        fingertip_x = obs[4] - obs[8]
        fingertip_y = obs[5] - obs[9]
        
        # Inject the updated moving targets directly into the observation array
        obs[4] = target_x
        obs[5] = target_y
        obs[8] = target_x - fingertip_x
        obs[9] = target_y - fingertip_y
        
        # 5. Calculate custom Dynamic Precision Reward
        joint_velocities = obs[6:8] 
        distance_to_target = np.linalg.norm(obs[8:10])
        self.episode_errors.append(distance_to_target)
        
        effort = np.sum(joint_velocities ** 2)
        
        sigma_0 = 0.025  
        lambd = 0.01    
        sigma_dynamic = sigma_0 + (lambd * effort)
        
        r_precision = np.exp(-(distance_to_target ** 2) / (sigma_dynamic ** 2))
        action_jerk = np.sum((action - self.prev_action) ** 2)
        
        # Combined balanced reward optimization
        custom_reward = (3.0 * r_precision) - (0.05 * action_jerk) - (0.005 * effort)
        
        self.prev_action = action.copy()
        
        # Enforce consistent episode lengths
        if self.t >= 50:
            truncated = True
            info["mean_episode_error"] = np.mean(self.episode_errors)
            
        return obs, custom_reward, terminated, truncated, info