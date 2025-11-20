import numpy as np

class StateCalculator:
    """
    Математическое ядро. Переводит координаты в "понятия" для робота.
    """

    def __init__(self):
        pass

    def calculate_state(self, robot_pos, robot_yaw, target_pos):
        """
        Args:
            robot_pos (array): [x, y] робота
            robot_yaw (float): Угол поворота робота (в радианах)
            target_pos (array): [x, y] цели
            
        Returns:
            (distance, angle) - Дистанция и Угол ошибки
        """
        
        # 1. Вектор до цели (dx, dy)
        # Это линия, соединяющая робота и цель
        diff = target_pos - robot_pos
        dx = diff[0]
        dy = diff[1]

        # 2. Дистанция (Теорема Пифагора)
        distance = np.linalg.norm(diff)

        # 3. Угол цели в "Мире" (Глобальный)
        # Какой угол у вектора цели относительно оси X карты?
        target_global_angle = np.arctan2(dy, dx)

        # 4. Угол ошибки (Относительный)
        # Куда цель смотрит ОТНОСИТЕЛЬНО носа робота?
        # Формула: (Куда надо смотреть) - (Куда я смотрю сейчас)
        angle_error = target_global_angle - robot_yaw

        # 5. Нормализация угла (Самая важная часть!)
        # Мы хотим угол строго от -PI до +PI (-180...180 градусов).
        # Если угол получился 350 градусов, превращаем его в -10.
        # Это позволяет роботу поворачивать по кратчайшему пути.
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi

        return distance, angle_error