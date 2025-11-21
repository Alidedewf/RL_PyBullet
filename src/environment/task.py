import pybullet as p
import numpy as np
import config

class Task:
    def __init__(self, client_id):
        self.cid = client_id
        # Создаем "Призрачную" цель (без коллизии для физики, но видимую)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1,0,0,1], physicsClientId=self.cid)
        # Отключаем коллизию (-1), чтобы робот мог проехать сквозь центр
        self.tid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, baseCollisionShapeIndex=-1, basePosition=[0,0,0], physicsClientId=self.cid)
        self.target_pos = np.array([0,0])

    def reset(self):
        r = np.random.uniform(1.0, config.MAX_TARGET_SPAWN_RADIUS)
        a = np.random.uniform(-np.pi, np.pi)
        self.target_pos = np.array([r * np.cos(a), r * np.sin(a)])
        p.resetBasePositionAndOrientation(self.tid, [self.target_pos[0], self.target_pos[1], 0.2], [0,0,0,1], physicsClientId=self.cid)
        return self.target_pos

    def get_target_position(self): return self.target_pos
    
    def check_goal_reached(self, robot_pos):
        return np.linalg.norm(robot_pos - self.target_pos) < config.DISTANCE_THRESHOLD