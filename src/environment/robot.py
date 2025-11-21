import pybullet as p
import numpy as np
import config

class Robot:
    def __init__(self, client_id):
        self.cid = client_id
        # Спавн на высоте 0.5, чтобы не застревал
        self.robot_id = p.loadURDF(config.ROBOT_URDF_PATH, [0,0,0.5], [0,0,0,1], physicsClientId=self.cid)

    def reset(self):
        p.resetBasePositionAndOrientation(self.robot_id, [0,0,0.5], [0,0,0,1], physicsClientId=self.cid)
        p.resetBaseVelocity(self.robot_id, [0,0,0], [0,0,0], physicsClientId=self.cid)

    def apply_action(self, action_id):
        vl, vr = config.ACTIONS_MAP[action_id]
        # Husky: Левые (2,4), Правые (3,5)
        for j in [2, 4]:
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, targetVelocity=vl, force=config.MAX_WHEEL_FORCE, physicsClientId=self.cid)
        for j in [3, 5]:
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, targetVelocity=vr, force=config.MAX_WHEEL_FORCE, physicsClientId=self.cid)

    def get_observation(self):
        pos, rot = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.cid)
        euler = p.getEulerFromQuaternion(rot)
        return np.array([pos[0], pos[1]]), euler[2]