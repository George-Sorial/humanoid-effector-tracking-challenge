import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.env import TrajectoryTrackingEnv

def smooth_curve(data, window_size=50):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():
    os.makedirs("results", exist_ok=True)
    
    # ----------------------------------------------------
    # PLOT 1: Generate the Training Progression Error Curve
    # ----------------------------------------------------
    print("Generating training error progression plots...")
    try:
        errors = np.load("results/training_errors.npy")
        if len(errors) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(errors, alpha=0.3, color='green', label='Raw Episode Error')
            smoothed = smooth_curve(errors, window_size=max(5, len(errors)//50))
            plt.plot(smoothed, color='darkgreen', linewidth=2, label='Smoothed Trend')
            plt.title("Action-Space Optimization: Mean Tracking Error Across Training")
            plt.xlabel("Completed Episodes")
            plt.ylabel("Mean Euclidean Error (meters)")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.savefig("results/training_error_curve.png", dpi=150)
            plt.close()
            print("Successfully saved: results/training_error_curve.png")
        else:
            print("Error log array is empty. Skip plotting error curve.")
    except FileNotFoundError:
        print("Could not find results/training_errors.npy. Run train.py first.")

    # ----------------------------------------------------
    # PLOT 2: Single Episode Detailed Spatial Analysis
    # ----------------------------------------------------
    print("Running evaluation deployment to plot physical tracking performance...")
    env = TrajectoryTrackingEnv()
    try:
        model = PPO.load("models/ppo_reacher_tracking")
    except FileNotFoundError:
        print("No trained weights file located. Evaluation fallback running random policy.")
        model = None

    obs, info = env.reset()
    
    target_trajectory = []
    fingertip_trajectory = []
    error_distances = []

    done = False
    while not done:
        # Recover absolute positions from the modified array tracking logic
        t_x, t_y = obs[4], obs[5]
        f_x, f_y = t_x - obs[8], t_y - obs[9]
        
        target_trajectory.append([t_x, t_y])
        fingertip_trajectory.append([f_x, f_y])
        error_distances.append(np.linalg.norm(obs[8:10]))
        
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    target_trajectory = np.array(target_trajectory)
    fingertip_trajectory = np.array(fingertip_trajectory)

    # Assemble the side-by-side performance layout graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left Panel: 2D Spatial Layout mapping
    ax1.plot(target_trajectory[:, 0], target_trajectory[:, 1], 'r--', linewidth=2, label='Target Circular Path')
    ax1.plot(fingertip_trajectory[:, 0], fingertip_trajectory[:, 1], 'b-', marker='o', alpha=0.7, label='Fingertip Path')
    ax1.scatter(target_trajectory[0, 0], target_trajectory[0, 1], color='red', marker='X', s=100, zorder=5, label='Start (Target)')
    ax1.scatter(fingertip_trajectory[0, 0], fingertip_trajectory[0, 1], color='blue', marker='X', s=100, zorder=5, label='Start (Robot)')
    ax1.set_title("2D Spatial Cartesian Tracking (MuJuCo)")
    ax1.set_xlabel("X coordinate (meters)")
    ax1.set_ylabel("Y coordinate (meters)")
    ax1.axis('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # Right Panel: Step-by-Step distance decay mapping
    ax2.plot(error_distances, 'r-', linewidth=2, marker='s', markersize=4)
    ax2.set_title("Tracking Error Distance Over Time")
    ax2.set_xlabel("Episode Timestep")
    ax2.set_ylabel("Euclidean Distance Error (meters)")
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig("results/tracking_performance.png", dpi=150)
    plt.close()
    print("Successfully saved: results/tracking_performance.png")

if __name__ == "__main__":
    main()