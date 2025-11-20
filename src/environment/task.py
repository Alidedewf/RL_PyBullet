import pybullet as p
import numpy as np
import config

class Task:
    """
    Отвечает *только* за задачу/цель:
    - Создание визуального объекта цели (сферы).
    - Спавн цели в случайной позиции (при reset).
    - Проверка, достигнута ли цель.
    """
    
    def __init__(self, client_id):
        """
        Создает визуальный объект цели в симуляции.
        
        Args:
            client_id (int): ID симулятора PyBullet.
        """
        self.client_id = client_id
        
        # 1. Создаем форму (визуальную)
        target_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=[1, 0, 0, 1] # Красный цвет
        )
        
        # 2. Создаем "тело" объекта (без физики, baseMass=0)
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual_shape,
            # !!! ИЗМЕНЕНИЕ: baseCollisionShapeIndex=-1 !!!
            # Это делает объект "призрачным" (нет физического тела),
            # робот сможет проехать сквозь него.
            baseCollisionShapeIndex=-1, 
            basePosition=[0, 0, 0],
            physicsClientId=self.client_id
        )
        
        self.target_pos_2d = np.array([0, 0])
        print(f"Task: Цель (сфера) создана (ID: {self.target_id}).")

    def reset(self):
        """
        Спавнит цель в новой случайной 2D-позиции.
        Возвращает 2D-координаты этой позиции.
        """
        # Случайная позиция в пределах радиуса
        radius = np.random.uniform(1.0, config.MAX_TARGET_SPAWN_RADIUS)
        angle = np.random.uniform(-np.pi, np.pi)
        
        target_x = radius * np.cos(angle)
        target_y = radius * np.sin(angle)
        
        self.target_pos_2d = np.array([target_x, target_y])
        
        # Перемещаем объект цели (на высоте 0.1)
        p.resetBasePositionAndOrientation(
            self.target_id, 
            [target_x, target_y, 0.1], 
            [0, 0, 0, 1],
            physicsClientId=self.client_id
        )
        
        return self.target_pos_2d

    def get_target_position(self):
        """ Возвращает текущую 2D-позицию цели. """
        return self.target_pos_2d
        
    def check_goal_reached(self, robot_pos_2d):
        """
        Проверяет, находится ли робот достаточно близко к цели.
        
        Args:
            robot_pos_2d (np.array): 2D-координаты робота.
            
        Returns:
            bool: True, если цель достигнута.
        """
        # Считаем 2D-дистанцию
        distance = np.linalg.norm(robot_pos_2d - self.target_pos_2d)
        
        # Сравниваем с порогом из конфига
        return distance < config.DISTANCE_THRESHOLD