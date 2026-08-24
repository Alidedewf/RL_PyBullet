import os
import time
from stable_baselines3 import PPO
from robot_env import RobotArmEnv

LOG_DIR = "./logs/robot_arm_visual_ppo"
MODEL_PATH = os.path.join(LOG_DIR, "final_model.zip")
N_EVAL_EPISODES = 50
TOP_K = 10

def evaluate_episodes(model, n_episodes):
    print(f"Evaluating {n_episodes} episodes (no GUI)...")
    results = []
    for i in range(n_episodes):
        seed = 1000 + i
        env = RobotArmEnv(render_mode='rgb_array', image_size=64, frame_skip=4, frame_stacks=4)
        env.set_seed(seed)
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        env.close()
        print(f"Episode {i+1}/{n_episodes} seed={seed} reward={total_reward:.3f}")
        results.append((seed, total_reward))
    return results

def replay_best_episodes(model, best_seeds):
    print(f"Replaying top {len(best_seeds)} episodes with GUI...")
    env = RobotArmEnv(render_mode='human', image_size=64, frame_skip=4, frame_stacks=4)
    try:
        for i, (seed, rew) in enumerate(best_seeds):
            print(f"Replay {i+1}/{len(best_seeds)} seed={seed} expected_reward={rew:.2f}")
            env.set_seed(seed)
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                time.sleep(1.0 / 60.0)
            time.sleep(0.6)
    except KeyboardInterrupt:
        print("Replay interrupted.")
    finally:
        env.close()

if __name__ == "__main__":
    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    results = evaluate_episodes(model, N_EVAL_EPISODES)
    results.sort(key=lambda x: x[1], reverse=True)
    best = results[:TOP_K]
    print("\nTop episodes:")
    for seed, rew in best:
        print(f"Seed={seed} reward={rew:.3f}")
    replay_best_episodes(model, best)