# environment.py
import pybullet as p
import pybullet_data

class SimpleEnv:
    def __init__(self, render=False):
        self.target_pos_x = 0.0  
        self.agent_start_pos = [5.0, 0, 0.5] 
        self.target_start_pos = [self.target_pos_x, 0, 0.5] 
        self.max_steps = 500
        self.step_counter = 0
        self.success_threshold = 0.2
        self.action_space = 2 # 0: влево, 1: вправо
        self.force_magnitude = 50.0

        if render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0/240.0, physicsClientId=self.client)

        self.planeId = p.loadURDF("plane.urdf", physicsClientId=self.client)
        
        agent_col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], physicsClientId=self.client)
        self.agentId = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=agent_col_shape,
                                         basePosition=self.agent_start_pos, physicsClientId=self.client)
        
        target_col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2, physicsClientId=self.client)
        target_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 0, 0, 1], physicsClientId=self.client)
        self.targetId = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=target_col_shape,
                                           baseVisualShapeIndex=target_visual_shape,
                                           basePosition=self.target_start_pos, physicsClientId=self.client)

    def get_state(self):
        pos, _ = p.getBasePositionAndOrientation(self.agentId, physicsClientId=self.client)
        return pos[0]

    def get_distance_to_target(self):
        agent_x = self.get_state()
        return abs(agent_x - self.target_pos_x)

    def reset(self):
        p.resetBasePositionAndOrientation(self.agentId, self.agent_start_pos, [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.agentId, [0, 0, 0], [0, 0, 0], physicsClientId=self.client) 
        self.step_counter = 0
        self.last_distance = self.get_distance_to_target()
        return self.get_state()

    def step(self, action):
        if action == 0:
            force = [-self.force_magnitude, 0, 0]
        elif action == 1:
            force = [self.force_magnitude, 0, 0]
        else:
            raise ValueError(f"Неверное действие: {action}")

        p.applyExternalForce(self.agentId, -1, force, [0, 0, 0], p.LINK_FRAME, physicsClientId=self.client)
        p.stepSimulation(physicsClientId=self.client)
        self.step_counter += 1

        new_state = self.get_state()
        new_distance = self.get_distance_to_target()

        reward = self.last_distance - new_distance
        self.last_distance = new_distance
        reward -= 0.1  # штраф за существование
        done = False
        if new_distance < self.success_threshold:
            reward += 100
            done = True
        elif self.step_counter >= self.max_steps:
            reward -= 10
            done = True

        return new_state, reward, done

    def close(self):
        p.disconnect(physicsClientId=self.client)