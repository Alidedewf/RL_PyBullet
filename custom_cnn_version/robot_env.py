import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
from collections import deque
import cv2 # Для Grayscale и Resizing

class RobotArmEnv(gym.Env):
    def __init__(self, render_mode='rgb_array', image_size=64, frame_skip=8):
        super(RobotArmEnv, self).__init__()
        
        # --- Параметры Среды ---
        self.image_size = image_size
        self.frame_skip = frame_skip
        self.max_steps = 200 # Максимальное количество шагов в эпизоде
        self.current_step = 0
        
        # --- PyBullet Setup ---
        self.physicsClient = p.connect(p.GUI if render_mode == 'human' else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient)
        
        # --- Загрузка Объектов ---
        self.planeId = p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)
        # Загрузка робота Panda
        self.robotId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=self.physicsClient)
        # Получение индекса концевого эффектора (gripper)
        self.end_effector_index = 11
        
        # Загрузка целевого объекта (маленький куб)
        self.target_object_id = p.loadURDF("cube.urdf", globalScaling=0.08, physicsClientId=self.physicsClient) 
        
        # Установка начальных положений джоинтов (для стабильности)
        self.initial_joint_positions = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
        for i in range(7):
            p.resetJointState(self.robotId, i, self.initial_joint_positions[i], physicsClientId=self.physicsClient)

        # --- Пространство Наблюдений (Observation Space) ---
        # 1. Пиксели: Grayscale 64x64x1 (C=1)
        pixel_shape = (image_size, image_size, 1)
        # 2. Проприоцепция: 7 углов джоинтов Panda
        proprio_shape = (7,) 
        
        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(low=0, high=255, shape=pixel_shape, dtype=np.uint8),
            "proprioception": spaces.Box(low=-np.pi, high=np.pi, shape=proprio_shape, dtype=np.float32)
        })
        
        # --- Пространство Действий (Action Space) ---
        # Delta X, Delta Y, Delta Z для концевого эффектора
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # --- Настройка Камеры (Eye-to-hand: над столом) ---
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.5, 0, 0.2], # Центр стола
            distance=0.7, 
            yaw=90, 
            pitch=-45, 
            roll=0, 
            upAxisIndex=2,
            physicsClientId=self.physicsClient
        )
        self.projMatrix = p.computeProjectionMatrixFOV(
            fov=60, 
            aspect=1.0, 
            nearVal=0.1, 
            farVal=10.0,
            physicsClientId=self.physicsClient
        )

    # --- Вспомогательные Функции ---

    def _get_observation(self):
        # 1. Получение изображения
        # Используем TinyRenderer всегда, так как он быстрее для маленьких изображений на CPU и не тормозит основной поток
        
        img_arr = p.getCameraImage(
            width=self.image_size,
            height=self.image_size,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projMatrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.physicsClient
        )
        
        if img_arr is None or img_arr[2] is None:
             # Fallback if rendering fails
             gray_img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        else:
            rgb_img = np.reshape(img_arr[2], (self.image_size, self.image_size, 4))[:, :, :3]
            # Преобразование в Grayscale (оптимизация)
            gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            
        pixels = np.expand_dims(gray_img, axis=-1) # Добавляем канал C=1

        # 2. Получение проприоцепции (углы первых 7 джоинтов Panda)
        joint_states = p.getJointStates(self.robotId, range(7), physicsClientId=self.physicsClient)
        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)

        return {
            "pixels": pixels,
            "proprioception": joint_positions
        }

    def _apply_action(self, action):
        # Преобразование относительного действия [-1, 1] в абсолютное смещение (например, +/- 1 см)
        max_delta = 0.01 
        delta_pos = action * max_delta
        
        # Получение текущей позиции схвата
        link_state = p.getLinkState(self.robotId, self.end_effector_index, physicsClientId=self.physicsClient)
        current_pos = np.array(link_state[0])
        
        # Вычисление целевой позиции
        target_pos = current_pos + delta_pos
        
        # Использование инверсной кинематики (IK) для вычисления углов джоинтов
        # Оставляем ориентацию схвата фиксированной для простоты (например, направленной вниз)
        orientation = p.getQuaternionFromEuler([0, -np.pi, 0]) 
        
        target_joints = p.calculateInverseKinematics(
            self.robotId,
            self.end_effector_index,
            target_pos.tolist(),
            orientation,
            maxNumIterations=100,
            physicsClientId=self.physicsClient
        )
        
        # Применение углов джоинтов
        for i in range(7):
            p.setJointMotorControl2(
                bodyUniqueId=self.robotId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_joints[i],
                force=500, # Сила для перемещения
                physicsClientId=self.physicsClient
            )

    def step(self, action):
        # --- 1. Применение Действия и Frame Skipping (Требование 3) ---
        self._apply_action(action)
        
        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.physicsClient)
        
        # --- 2. Получение Информации ---
        obs = self._get_observation()
        self.current_step += 1
        
        # --- 3. Расчет Награды (Reward Function) ---
        link_state = p.getLinkState(self.robotId, self.end_effector_index, physicsClientId=self.physicsClient)
        tool_pos = np.array(link_state[0])
        
        obj_pos, _ = p.getBasePositionAndOrientation(self.target_object_id, physicsClientId=self.physicsClient)
        obj_pos = np.array(obj_pos)
        
        # Расстояние до цели
        distance = np.linalg.norm(tool_pos - obj_pos)
        
        # Контакт (Sparse Reward) - проверяем контакт между схватом и объектом
        contact_points = p.getContactPoints(self.robotId, self.target_object_id, self.end_effector_index, physicsClientId=self.physicsClient)
        is_contact = len(contact_points) > 0
        
        if is_contact:
            print("CONTACT! +100")
        
        # Веса награды (Требование 5)
        # Уменьшили штраф за расстояние (w1: 10.0 -> 1.0), чтобы общий итог при успехе был положительным
        w1, w2, w3 = 1.0, 100.0, 0.01
        
        reward = - w1 * distance + w2 * is_contact - w3
        
        # --- 4. Проверка Завершения Эпизода ---
        terminated = is_contact or self.current_step >= self.max_steps
        truncated = False # В gymnasium 0.29+
        info = {"distance": distance}
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if hasattr(self, 'next_seed') and self.next_seed is not None:
            seed = self.next_seed
            self.next_seed = None
            
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Сброс робота в начальное положение
        for i in range(7):
            p.resetJointState(self.robotId, i, self.initial_joint_positions[i], physicsClientId=self.physicsClient)
        
        # Сброс целевого объекта в случайное место (в пределах рабочей зоны)
        # Рабочая зона X: [0.3, 0.7], Y: [-0.3, 0.3], Z: [0.0]
        rand_x = self.np_random.uniform(low=0.3, high=0.7)
        rand_y = self.np_random.uniform(low=-0.3, high=0.3)
        p.resetBasePositionAndOrientation(self.target_object_id, [rand_x, rand_y, 0.02], [0, 0, 0, 1], physicsClientId=self.physicsClient)

        # Сброс симулятора и получение первого наблюдения
        for _ in range(100): # Даем симулятору стабилизироваться
            p.stepSimulation(physicsClientId=self.physicsClient)
            
        initial_obs = self._get_observation()
        info = {}
        
        return initial_obs, info

    def render(self):
        # PyBullet GUI handles rendering automatically
        pass

    def set_seed(self, seed):
        self.next_seed = seed

    def close(self):
        if p.isConnected(physicsClientId=self.physicsClient):
            p.disconnect(physicsClientId=self.physicsClient)

# Регистрация среды для удобства
gym.envs.registration.register(
    id='RobotArmVisual-v0',
    entry_point='robot_env:RobotArmEnv',
    max_episode_steps=200,
    kwargs={'render_mode': 'rgb_array', 'image_size': 64, 'frame_skip': 8}
)