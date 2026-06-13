import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env_3d import TrajectoryTracking3DEnv

class ErrorLoggingCallback(BaseCallback):
    def __init__(self, check_freq=1, log_dir="results", verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.history = []
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "mean_episode_error" in info:
                self.history.append(info["mean_episode_error"])
                
        if self.n_calls % 500 == 0:
            np.save(os.path.join(self.log_dir, "training_errors.npy"), np.array(self.history))
            
        return True

def main():
    print("Initializing 3D PyBullet environment...")
    raw_env = TrajectoryTracking3DEnv(render_mode=None) # Fast mode
    env = Monitor(raw_env)
    env = DummyVecEnv([lambda: env])

    print("Setting up the PPO neural network agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003, # Slightly lower for 7-DOF stability
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        device="cpu"
    )

    os.makedirs("models", exist_ok=True)
    error_callback = ErrorLoggingCallback(log_dir="results")
    
    print("Starting training loop...")
    model.learn(total_timesteps=500000, callback=error_callback)
    
    model.save("models/ppo_kuka_tracking_3d")
    np.save("results/training_errors.npy", np.array(error_callback.history))
    print("Training complete! Model saved.")

if __name__ == "__main__":
    main()