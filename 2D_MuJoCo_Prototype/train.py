import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from src.env import TrajectoryTrackingEnv

class ErrorLoggingCallback(BaseCallback):
    def __init__(self, check_freq=1, log_dir="results", verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.history = []
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if an episode finished in any of the vectorized environments
        for info in self.locals.get("infos", []):
            if "mean_episode_error" in info:
                self.history.append(info["mean_episode_error"])
                
        # Periodically save error array data to disk
        if self.n_calls % 500 == 0:
            np.save(os.path.join(self.log_dir, "training_errors.npy"), np.array(self.history))
            
        return True

def main():
    print("Initializing your custom trajectory tracking environment...")
    raw_env = TrajectoryTrackingEnv()
    env = Monitor(raw_env)
    env = DummyVecEnv([lambda: env])

    print("Setting up the PPO neural network agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0005,
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

    print("Starting training loop with dynamic tracking logs...")
    os.makedirs("models", exist_ok=True)
    error_callback = ErrorLoggingCallback(log_dir="results")
    
    # Train for a sufficient duration to track the learning progression
    model.learn(total_timesteps=300000, callback=error_callback)
    
    # Save the final model weights
    model.save("models/ppo_reacher_tracking")
    np.save("results/training_errors.npy", np.array(error_callback.history))
    print("Training complete! Model saved to models/ppo_reacher_tracking")

if __name__ == "__main__":
    main()