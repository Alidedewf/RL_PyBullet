import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import os
import torch as th

from robot_env import RobotArmEnv
# Import MobileNet Extractor instead of custom CNN
from mobilenet_extractor import MobileNetCombinedExtractor 

# --- Конфигурация Обучения ---
LOG_DIR = "./logs/robot_arm_mobilenet_ppo"
# --- Конфигурация Обучения ---
LOG_DIR = "./logs/robot_arm_mobilenet_ppo"
TOTAL_TIMESTEPS = 100_000
FRAME_STACKS = 4 # Требование 3: Стек из 4 последних кадров
FEATURES_DIM = 512 # Размер выхода (игнорируется, так как определяется в extractor)

if __name__ == '__main__':
    # 1. Создание и Оборачивание Среды
    
    env_id = "RobotArmVisual-v0"
    n_envs = 4 # Уменьшаем до 4 для стабильности на Mac M2 (MobileNet on CPU)
    vec_env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=DummyVecEnv)
    
    # Frame Stacking: Обертка для добавления динамики (Требование 3)
    vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACKS, channels_order='last')
    
    # VecTransposeImage: Преобразует (H, W, C) -> (C, H, W) для PyTorch
    vec_env = VecTransposeImage(vec_env)

    # 2. Создание Модели PPO
    policy_kwargs = dict(
        features_extractor_class=MobileNetCombinedExtractor, # Используем MobileNet
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
    )
    
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
    
    # 3. Callbacks
    # Создаем eval_env правильно (с TransposeImage)
    eval_env = DummyVecEnv([lambda: RobotArmEnv()])
    eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACKS, channels_order='last')
    eval_env = VecTransposeImage(eval_env)
    
    # Сохранение модели каждые 20000 шагов
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=LOG_DIR, name_prefix="rl_model")
    
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