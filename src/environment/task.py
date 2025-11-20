import pybullet as p
import numpy as np
import config

class Task:
    def __init__(self, client_id):
        self.client_id = client_id
        
        # Визуал (Красный шар)
        target_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.2, # Чуть побольше, чтобы заметнее
            rgbaColor=[1, 0, 0, 1]
        )
        
        # Физика (Коллизия)
        target_col_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.2
        )
        
        # Создаем тело. Теперь оно ТВЕРДОЕ (есть CollisionShape).
        self.target_id = p.createMultiBody(
            baseMass=100, # Тяжелый шар, чтобы робот его не упинал в космос
            baseVisualShapeIndex=target_visual_shape,
            baseCollisionShapeIndex=target_col_shape, # <-- ВКЛЮЧИЛИ КОЛЛИЗИЮ
            basePosition=[0, 0, 0],
            physicsClientId=self.client_id
        )
        
        self.target_pos_2d = np.array([0, 0])

    def reset(self):
        radius = np.random.uniform(1.0, config.MAX_TARGET_SPAWN_RADIUS)
        angle = np.random.uniform(-np.pi, np.pi)
        
        target_x = radius * np.cos(angle)
        target_y = radius * np.sin(angle)
        
        self.target_pos_2d = np.array([target_x, target_y])
        
        p.resetBasePositionAndOrientation(
            self.target_id, 
            [target_x, target_y, 0.2], 
            [0, 0, 0, 1],
            physicsClientId=self.client_id
        )
        
        return self.target_pos_2d

    def get_target_position(self):
        return self.target_pos_2d
        
    def check_goal_reached(self, robot_pos_2d):
        distance = np.linalg.norm(robot_pos_2d - self.target_pos_2d)
        return distance < config.DISTANCE_THRESHOLD