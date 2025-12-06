
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
import os
import torch as th
from robot_env import RobotArmEnv
from custom_cnn import CustomCombinedExtractor
import time

from stable_baselines3.common.callbacks import CheckpointCallback

# --- Конфигурация Студента ---
TEACHER_PATH = "./logs/teacher_ppo_coords/teacher_model"
LOG_DIR = "./logs/student_ppo_curriculum"
TOTAL_TIMESTEPS = 300_000
FRAME_STACKS = 4

def make_env():
    env = RobotArmEnv(obs_type='pixels')
    return env

if __name__ == '__main__':
    # 1. Создание Среды (Pixels Mode)
    # Используем ту же обертку, что и в Варианте А (быструю)
    vec_env = DummyVecEnv([make_env for _ in range(4)]) # 4 среды для скорости
    vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACKS, channels_order='last')
    vec_env = VecTransposeImage(vec_env) # HWC -> CHW

    # 2. Инициализация Студента (MultiInputPolicy с Custom CNN)
    print("Инициализация Студента...")
    student_model = PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs={
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {"features_dim": 256}, 
            "net_arch": [64, 64] 
        },
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto"
    )

    # 3. Загрузка Учителя
    print(f"Загрузка Учителя из {TEACHER_PATH}.zip...")
    if not os.path.exists(TEACHER_PATH + ".zip"):
        print("Ошибка: Файл учителя не найден! Сначала запустите train_teacher.py")
        exit(1)
        
    teacher_model = PPO.load(TEACHER_PATH, device='cpu')

    # 4. Перенос Знаний (Weight Transfer / Neural Surgery)
    print("--- НАЧАЛО ХИРУРГИИ МОЗГА (Transfer Weights) ---")
    
    student_dict = student_model.policy.state_dict()
    teacher_dict = teacher_model.policy.state_dict()
    
    transferred_layers = []
    
    # Перебираем веса учителя и пытаемся вставить их студенту
    for key, param in teacher_dict.items():
        if key in student_dict:
            student_param = student_dict[key]
            if student_param.shape == param.shape:
                student_dict[key].copy_(param)
                transferred_layers.append(key)
    
    print(f"Пересажено слоев: {len(transferred_layers)}")
    for layer in transferred_layers:
        print(f"  - {layer}")
        
    # Загружаем обновленный dict обратно в модель
    student_model.policy.load_state_dict(student_dict)
    print("--- ХИРУРГИЯ ЗАВЕРШЕНА УСПЕШНО ---")

    # 5. Обучение Студента (Fine-tuning)
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=LOG_DIR, name_prefix="student_model")
    
    print(f"Начинаем дообучение Студента ({TOTAL_TIMESTEPS} шагов)...")
    start_time = time.time()
    student_model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    end_time = time.time()
    
    print(f"--- Обучение Студента завершено за {end_time - start_time:.2f} сек ---")

    # 6. Сохранение
    save_path = os.path.join(LOG_DIR, "final_student_model")
    student_model.save(save_path)
    print(f"Модель Студента сохранена в {save_path}.zip")
