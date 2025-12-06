import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import os
from robot_env import RobotArmEnv
from custom_cnn import CustomCombinedExtractor
import pybullet as p
import numpy as np
import time

# --- Конфигурация ---
LOG_DIR = "./logs/robot_arm_visual_ppo"
MODEL_PATH = os.path.join(LOG_DIR, "final_model.zip")
FRAME_STACKS = 4
N_EVAL_EPISODES = 50
TOP_K = 10

def make_env(render_mode='rgb_array'):
    env = DummyVecEnv([lambda: RobotArmEnv(render_mode=render_mode)])
    env = VecFrameStack(env, n_stack=FRAME_STACKS, channels_order='last')
    return env

def evaluate_episodes(model, n_episodes):
    print(f"--- Оценка {n_episodes} эпизодов (без графики) ---")
    env = make_env(render_mode='rgb_array')
    results = []
    
    try:
        for i in range(n_episodes):
            seed = 1000 + i
            # Установка сида для следующего сброса
            env.env_method("set_seed", seed)
            
            obs = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward[0]
            
            print(f"Эпизод {i+1}/{n_episodes} (Seed {seed}): Награда = {total_reward:.2f}")
            results.append((seed, total_reward))
            
    finally:
        env.close()
        
    return results

def replay_best_episodes(model, best_seeds):
    print(f"\n--- Показ {len(best_seeds)} лучших эпизодов (с графикой) ---")
    env = make_env(render_mode='human')
    
    try:
        for i, (seed, reward) in enumerate(best_seeds):
            print(f"Показ {i+1}/{len(best_seeds)}: Seed {seed}, Ожидаемая награда: {reward:.2f}")
            
            env.env_method("set_seed", seed)
            obs = env.reset()
            
            # Настройка камеры для лучшего вида (опционально)
            # p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0.5, 0, 0])
            # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Скрыть интерфейс PyBullet
            
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = env.step(action)
                # env.render() # PyBullet GUI handles rendering
            
            time.sleep(0.5) # Небольшая пауза между эпизодами
                
    except KeyboardInterrupt:
        print("Просмотр прерван.")
    except p.error:
        print("Окно симуляции закрыто.")
    finally:
        try:
            env.close()
        except:
            pass

if __name__ == '__main__':
    print(f"Загрузка модели из {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)

    # 1. Поиск лучших эпизодов
    results = evaluate_episodes(model, N_EVAL_EPISODES)
    
    # Сортировка по награде (по убыванию)
    results.sort(key=lambda x: x[1], reverse=True)
    best_results = results[:TOP_K]
    
    print("\nТоп 10 эпизодов:")
    for seed, reward in best_results:
        print(f"Seed {seed}: {reward:.2f}")

    # 2. Проигрывание лучших
    replay_best_episodes(model, best_results)
