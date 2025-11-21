import pybullet as p
import pybullet_data
import time
import config

class SimulationManager:
    def __init__(self):
        self.client_id = -1
        if config.GUI_MODE:
            self.client_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
            p.resetDebugVisualizerCamera(5, 0, -40, [0,0,0], physicsClientId=self.client_id)
        else:
            self.client_id = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setRealTimeSimulation(0, physicsClientId=self.client_id)
        p.setTimeStep(config.SIM_TIMESTEP, physicsClientId=self.client_id)
        
        try:
            self.plane = p.loadURDF(config.PLANE_URDF_PATH, physicsClientId=self.client_id)
            p.changeDynamics(self.plane, -1, lateralFriction=1.0, physicsClientId=self.client_id)
        except: pass

    def step(self):
        p.stepSimulation(physicsClientId=self.client_id)
        if config.GUI_MODE: time.sleep(config.SIM_TIMESTEP)

    def close(self):
        if self.client_id != -1: p.disconnect(self.client_id)
    
    def get_client_id(self): return self.client_id