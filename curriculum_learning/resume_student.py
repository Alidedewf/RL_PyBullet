
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
import os
from robot_env import RobotArmEnv
from custom_cnn import CustomCombinedExtractor
import time

# --- Конфигурация Рестарта ---
# Используем последний чекпоинт
CHECKPOINT_PATH = "./logs/student_ppo_curriculum/student_model_80000_steps.zip"
LOG_DIR = "./logs/student_ppo_curriculum_resumed"
# Осталось обучить: 300,000 (Цель) - 80,000 (Сделано) = 220,000
TOTAL_TIMESTEPS = 220_000 
FRAME_STACKS = 4

def make_env():
    env = RobotArmEnv(obs_type='pixels')
    return env

if __name__ == '__main__':
    print(f"--- ВОЗОБНОВЛЕНИЕ ОБУЧЕНИЯ (RESUME) ---")
    print(f"Загрузка чекпоинта: {CHECKPOINT_PATH}")
    
    # 1. Создание Среды (как раньше)
    vec_env = DummyVecEnv([make_env for _ in range(4)])
    vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACKS, channels_order='last')
    vec_env = VecTransposeImage(vec_env)

    # 2. Загрузка Модели
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Ошибка! Чекпоинт не найден: {CHECKPOINT_PATH}")
        exit(1)

    model = PPO.load(CHECKPOINT_PATH, env=vec_env, device='auto', verbose=1, tensorboard_log=LOG_DIR)

    # 3. Продолжение Обучения с progress_bar=True
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=LOG_DIR, name_prefix="student_resumed")
    
    print(f"Старт дообучения на {TOTAL_TIMESTEPS} шагов с прогресс-баром...")
    start_time = time.time()
    
    # Включаем progress_bar=True
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, reset_num_timesteps=False, progress_bar=True)
    
    end_time = time.time()
    print(f"--- Обучение завершено за {end_time - start_time:.2f} сек ---")

    # 4. Финальное сохранение
    save_path = os.path.join(LOG_DIR, "final_student_model_resumed")
    model.save(save_path)
    print(f"Модель сохранена в {save_path}.zip")
