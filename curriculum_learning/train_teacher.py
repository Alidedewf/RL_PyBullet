
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from robot_env import RobotArmEnv
import time

# --- Конфигурация Обучения Учителя ---
LOG_DIR = "./logs/teacher_ppo_coords"
TOTAL_TIMESTEPS = 50_000

if __name__ == '__main__':
    # 1. Создание Среды (Coords Mode)
    # Используем obs_type='coords' для обучения на координатах
    env_kwargs = {'obs_type': 'coords'}
    
    # Векторизация среды (4 параллельных среды для скорости, хотя MlpPolicy и так быстрый)
    vec_env = make_vec_env(RobotArmEnv, n_envs=8, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv)

    # 2. Инициализация Модели (MlpPolicy - простая полносвязная сеть)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto" # CPU или GPU, MlpPolicy легкий
    )

    print(f"--- Начинаем обучение Учителя (на координатах) ---")
    print(f"Цель: {TOTAL_TIMESTEPS} шагов")

    # 3. Обучение
    start_time = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    end_time = time.time()
    
    print(f"--- Обучение завершено за {end_time - start_time:.2f} сек ---")

    # 4. Сохранение
    save_path = os.path.join(LOG_DIR, "teacher_model")
    model.save(save_path)
    print(f"Модель Учителя сохранена в {save_path}.zip")
