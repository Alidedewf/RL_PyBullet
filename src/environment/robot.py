import pybullet as p
import numpy as np
import config

class Robot:
    """
    Отвечает *только* за робота:
    - Загрузка URDF.
    - Сброс позиции (reset).
    - Управление моторами (apply_action).
    - Получение своего состояния (get_observation).
    """
    
    def __init__(self, client_id):
        self.client_id = client_id
        
        # Начальная позиция робота [x, y, z]
        start_pos = [0, 0, 1] 
        start_orientation = p.getQuaternionFromEuler([0, 0, 0]) 
        
        self.robot_id = p.loadURDF(
            config.ROBOT_URDF_PATH, 
            start_pos, 
            start_orientation,
            physicsClientId=self.client_id
        )
        print(f"Robot: R2D2 загружен (ID: {self.robot_id}).")

    def reset(self):
        """
        Сбрасывает позицию и скорость робота в начало координат.
        """
        p.resetBasePositionAndOrientation(
            self.robot_id, 
            [0, 0, 1], 
            [0, 0, 0, 1], 
            physicsClientId=self.client_id
        )
        p.resetBaseVelocity(
            self.robot_id, 
            [0, 0, 0], 
            [0, 0, 0], 
            physicsClientId=self.client_id
        )

    def apply_action(self, action_id):
        # Получаем скорости (v_left, v_right) из нашего конфига
        v_left, v_right = config.ACTIONS_MAP[action_id]
        
        # Отправляем команду левому колесу
        for joint in [2, 4]: 
            p.setJointMotorControl2(
                self.robot_id, joint, p.VELOCITY_CONTROL,
                targetVelocity=v_left, force=config.MAX_WHEEL_FORCE,
                physicsClientId=self.client_id
            )
            
        # Крутим ПРАВЫЕ
        for joint in [3, 5]:
            p.setJointMotorControl2(
                self.robot_id, joint, p.VELOCITY_CONTROL,
                targetVelocity=v_right, force=config.MAX_WHEEL_FORCE,
                physicsClientId=self.client_id
            )

    def get_observation(self):
        # 1. Позиция и ориентация робота
        pos, orient_q = p.getBasePositionAndOrientation(
            self.robot_id, 
            physicsClientId=self.client_id
        )
        
        # 2. Конвертируем кватернион ориентации в углы Эйлера
        orient_euler = p.getEulerFromQuaternion(orient_q)
        
        # 3. Выбираем то, что нам нужно
        robot_pos_2d = np.array([pos[0], pos[1]])
        robot_yaw = orient_euler[2] 
        
        return robot_pos_2d, robot_yaw