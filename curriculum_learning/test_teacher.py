
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from robot_env import RobotArmEnv
import pybullet as p
import time

# --- Конфигурация ---
LOG_DIR = "./logs/teacher_ppo_coords"
MODEL_PATH = os.path.join(LOG_DIR, "teacher_model.zip")
N_EVAL_EPISODES = 5

def make_env(render_mode='human'):
    # obs_type='coords' !!!
    env = DummyVecEnv([lambda: RobotArmEnv(render_mode=render_mode, obs_type='coords')])
    return env

def replay_teacher():
    print(f"--- Проверка Учителя (Координаты) ---")
    print(f"Загрузка: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print("Файл модели не найден!")
        return

    model = PPO.load(MODEL_PATH)
    env = make_env(render_mode='human')
    
    try:
        for i in range(N_EVAL_EPISODES):
            print(f"Эпизод {i+1}")
            obs = env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = env.step(action)
                time.sleep(1/60) 
                
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == '__main__':
    replay_teacher()
