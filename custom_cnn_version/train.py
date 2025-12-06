import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import os
import torch as th

# Импорт вашей среды и архитектуры
from robot_env import RobotArmEnv # Убедитесь, что робот_env.py находится в той же папке
from custom_cnn import CustomCombinedExtractor 

# --- Конфигурация Обучения ---
LOG_DIR = "./logs/robot_arm_visual_ppo"
TOTAL_TIMESTEPS = 300_000
FRAME_STACKS = 4 # Требование 3: Стек из 4 последних кадров
FEATURES_DIM = 512 # Размер выхода CNN перед объединением

if __name__ == '__main__':
    # 1. Создание и Оборачивание Среды
    # Использование make_vec_env для создания векторизованной среды
    env_id = "RobotArmVisual-v0"
    n_envs = 16 # Используем 16 параллельных среды для ускорения сбора данных
    vec_env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=DummyVecEnv)
    
    # Frame Stacking: Обертка для добавления динамики (Требование 3)
    # Note: VecFrameStack ожидает Grayscale (C=1) и объединяет их в (C*FRAME_STACKS, H, W)
    vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACKS, channels_order='last')

    # 2. Настройка Пользовательской Архитектуры (Vision Module)
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
        # Настройка MLP Policy Head (на входе FEATURES_DIM + 64 (proprioception))
        net_arch=[dict(pi=[256, 128], vf=[256, 128])] 
    )

    # 3. Инициализация Модели PPO
    model = PPO(
        "MultiInputPolicy", # Используется для Dict Space
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2048, # Размер буфера сбора данных
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log=LOG_DIR,
        device="cuda" if th.cuda.is_available() else "cpu"
    )

    # 4. Обратные Вызовы (Callbacks)
    # Сохранение лучшей модели и чекпоинтов
    eval_env = DummyVecEnv([lambda: RobotArmEnv()])
    eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACKS, channels_order='last')
    
    # Сохранение модели каждые 50000 шагов
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=LOG_DIR, name_prefix="rl_model")
    
    # Оценка модели каждые 10000 шагов
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=os.path.join(LOG_DIR, "best_model"), 
        log_path=LOG_DIR, 
        eval_freq=10000 // n_envs, 
        n_eval_episodes=5
    )

    # 5. Обучение
    print(f"--- Начинаем обучение. Логи в: {LOG_DIR} ---")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Обучение прервано пользователем.")
    
    # Сохранение финальной модели
    model.save(os.path.join(LOG_DIR, "final_model.zip"))
    print("--- Обучение завершено. ---")