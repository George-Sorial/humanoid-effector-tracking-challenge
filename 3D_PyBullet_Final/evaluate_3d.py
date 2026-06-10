import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env_3d import TrajectoryTracking3DEnv
import pybullet as p


def smooth_curve(data, window_size=50):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def draw_trajectory_preview(env, num_points=200):
    """Draw the full desired trajectory as a static red dotted path in PyBullet."""
    pts = [env._get_target(t) for t in range(num_points)]
    for i in range(len(pts) - 1):
        p.addUserDebugLine(
            pts[i].tolist(), pts[i + 1].tolist(),
            lineColorRGB=[1.0, 0.1, 0.1],
            lineWidth=1.5,
            lifeTime=0   # 0 = permanent
        )
    # Mark the start with a sphere-like cluster
    p.addUserDebugText("TARGET PATH", pts[0].tolist(),
                       textColorRGB=[1, 0, 0], textSize=1.2, lifeTime=0)


def main():
    os.makedirs("results", exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # PLOT 1: Training Curve
    # ──────────────────────────────────────────────────────────────────────────
    try:
        errors = np.load("results/training_errors.npy")
        if len(errors) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(errors, alpha=0.25, color='green', label='Raw Episode Error')
            smoothed = smooth_curve(errors, window_size=max(5, len(errors) // 50))
            ax1.plot(smoothed, color='darkgreen', linewidth=2, label='Smoothed Trend')
            ax1.set_title("Full Training History")
            ax1.set_xlabel("Completed Episodes")
            ax1.set_ylabel("Mean Euclidean Error (meters)")
            ax1.grid(True, linestyle="--", alpha=0.6)
            ax1.legend()

            cutoff = int(len(errors) * 0.2)
            late_errors = errors[-cutoff:]
            smoothed_late = smooth_curve(late_errors, window_size=max(5, len(late_errors) // 20))
            ax2.plot(late_errors, alpha=0.3, color='green', label='Raw Episode Error')
            ax2.plot(smoothed_late, color='darkgreen', linewidth=2, label='Smoothed Trend')
            final_mean = np.mean(late_errors)
            ax2.axhline(final_mean, color='red', ls='--', lw=1.5,
                        label=f'Converged mean: {final_mean*100:.2f} cm')
            ax2.set_title("Converged Performance (Final 20% of Training)")
            ax2.set_xlabel("Completed Episodes")
            ax2.set_ylabel("Mean Euclidean Error (meters)")
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.legend()

            plt.suptitle("PPO 3D End-Effector Tracking — Training Curve", fontsize=13)
            plt.tight_layout()
            plt.savefig("results/training_error_curve_3d.png", dpi=150)
            plt.close()
            print(f"Saved: results/training_error_curve_3d.png")
            print(f"Final converged mean error: {final_mean*100:.4f} cm")
    except FileNotFoundError:
        print("No training_errors.npy found. Run train_3d.py first.")

    # ──────────────────────────────────────────────────────────────────────────
    # PLOT 2: Single Episode Evaluation  (GUI stays open)
    # ──────────────────────────────────────────────────────────────────────────
    print("\nRunning evaluation deployment...")
    env = TrajectoryTracking3DEnv(render_mode="human")

    try:
        model = PPO.load("models/ppo_kuka_tracking_3d")
        print("Loaded trained model.")
    except FileNotFoundError:
        print("No trained weights found. Running random policy.")
        model = None

    obs, info = env.reset()

    # ── Draw the full desired trajectory immediately on reset ─────────────────
    draw_trajectory_preview(env, num_points=env.max_steps)

    target_traj, ee_traj, error_distances = [], [], []
    prev_ee = None
    done = False

    while not done:
        ee_pos     = obs[14:17]
        target_pos = obs[17:20]

        target_traj.append(target_pos.copy())
        ee_traj.append(ee_pos.copy())
        error_distances.append(float(np.linalg.norm(obs[20:23])))

        # ── Green EE trail ────────────────────────────────────────────────────
        if prev_ee is not None:
            p.addUserDebugLine(
                prev_ee.tolist(), ee_pos.tolist(),
                lineColorRGB=[0.1, 0.85, 0.1],
                lineWidth=3,
                lifeTime=0   # permanent — builds up the full path
            )

        # ── Live target cursor (bright yellow dot) ────────────────────────────
        p.addUserDebugText(
            "◆", target_pos.tolist(),
            textColorRGB=[1.0, 0.9, 0.0],
            textSize=1.0,
            lifeTime=0.08   # flickers along the path
        )

        prev_ee = ee_pos.copy()

        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        time.sleep(0.02)   # ~50 fps — slow enough to watch comfortably

    print("Episode complete. Keeping GUI open for 8 seconds...")
    time.sleep(8)          # hold so you can inspect / screenshot

    target_traj = np.array(target_traj)
    ee_traj     = np.array(ee_traj)
    errors_arr  = np.array(error_distances)

    # ── Terminal metrics (exclude cold-start first 30 steps) ─────────────────
    steady = errors_arr[30:170] # Focus on the steady-state portion of the episode
    print(f"\n── Evaluation Metrics (steady-state, steps 30–170) ──────")
    print(f"  Mean error  : {steady.mean()*100:.2f} cm")
    print(f"  RMSE        : {np.sqrt((steady**2).mean())*100:.2f} cm")
    print(f"  Max error   : {steady.max()*100:.2f} cm")
    print(f"  Min error   : {steady.min()*100:.2f} cm")
    print(f"  P95 error   : {np.percentile(steady, 95)*100:.2f} cm")
    print(f"─────────────────────────────────────────────────────────")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 6))

    # Panel 1: 3D spatial
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2],
             'r--', linewidth=2, label='Target Path', zorder=3)
    ax1.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2],
             color='#2196F3', linewidth=1.5, marker='o', markersize=2,
             alpha=0.85, label='EE Path')
    ax1.scatter(*target_traj[0], color='red',      marker='X', s=120, zorder=5)
    ax1.scatter(*ee_traj[0],     color='#2196F3',  marker='X', s=120, zorder=5)
    ax1.set_title("3D Cartesian Tracking\n(PyBullet Kuka iiwa)")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")
    ax1.legend(fontsize=8)

    # Panel 2: Error over time — full episode + steady-state mean
    steps = np.arange(len(errors_arr))
    ax2 = fig.add_subplot(132)
    ax2.plot(steps, errors_arr * 100, 'r-', linewidth=1.4, alpha=0.85)
    ax2.fill_between(steps, 0, errors_arr * 100, alpha=0.12, color='red')
    ax2.axvline(30, color='gray', ls=':', lw=1.2, label='Cold-start end (step 30)')
    ax2.axhline(steady.mean() * 100, color='darkred', ls='--', lw=1.5,
                label=f"Steady-state mean: {steady.mean()*100:.1f} cm")
    ax2.set_title("3D Tracking Error Over Time")
    ax2.set_xlabel("Episode Timestep")
    ax2.set_ylabel("Euclidean Error (cm)")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Panel 3: Per-axis breakdown
    ax3 = fig.add_subplot(133)
    ax3.plot(steps, (target_traj[:, 0] - ee_traj[:, 0]) * 100,
             color='#2196F3', lw=1.3, label='X error')
    ax3.plot(steps, (target_traj[:, 1] - ee_traj[:, 1]) * 100,
             color='#4CAF50', lw=1.3, label='Y error')
    ax3.plot(steps, (target_traj[:, 2] - ee_traj[:, 2]) * 100,
             color='#FF5722', lw=1.3, label='Z error')
    ax3.axhline(0, color='black', lw=0.8, ls='--')
    ax3.axvline(30, color='gray', ls=':', lw=1.2)
    ax3.set_title("Per-Axis Tracking Error")
    ax3.set_xlabel("Episode Timestep")
    ax3.set_ylabel("Error (cm)")
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("3D End-Effector Tracking — Evaluation Results", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/tracking_performance_3d.png", dpi=150, bbox_inches='tight')
    print("Saved: results/tracking_performance_3d.png")

    env.close()


if __name__ == "__main__":
    main()