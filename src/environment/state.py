import numpy as np

class StateCalculator:
    def calculate_state(self, r_pos, r_yaw, t_pos):
        diff = t_pos - r_pos
        dist = np.linalg.norm(diff)
        
        # Угол ошибки
        target_angle = np.arctan2(diff[1], diff[0])
        angle_err = target_angle - r_yaw
        
        # Нормализация (-PI, PI)
        while angle_err > np.pi: angle_err -= 2*np.pi
        while angle_err < -np.pi: angle_err += 2*np.pi
            
        return dist, angle_err