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
            p.resetDebugVisualizerCamera(
                cameraDistance=5, 
                cameraYaw=0, 
                cameraPitch=-40, 
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self.client_id
            )
        else:
            self.client_id = p.connect(p.DIRECT)
    
        
        # 2. Настройка симулятора
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), 
                                  physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setRealTimeSimulation(0, physicsClientId=self.client_id)
        p.setTimeStep(config.SIM_TIMESTEP, physicsClientId=self.client_id)
        
        p.loadURDF(config.PLANE_URDF_PATH, physicsClientId=self.client_id)
        self.plane_id = p.loadURDF(config.PLANE_URDF_PATH, physicsClientId=self.client_id)

        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, physicsClientId=self.client_id)


        print(f"SimulationManager: Успешно подключено (Client ID: {self.client_id}).")

    def step(self):
        """ Выполняет один шаг физики и делает паузу для GUI. """
        p.stepSimulation(physicsClientId=self.client_id)
        
        if config.GUI_MODE:
            time.sleep(config.SIM_TIMESTEP)

    def close(self):
        if self.client_id != -1 and p.isConnected(self.client_id):
            p.disconnect(self.client_id)
            print(f"SimulationManager: Отключено (Client ID: {self.client_id}).")
            self.client_id = -1
            
    def get_client_id(self):
        """ 
        Возвращает ID клиента. 
        """
        return self.client_id