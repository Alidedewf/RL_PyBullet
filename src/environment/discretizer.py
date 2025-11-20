import numpy as np
import config

class StateDiscretizer:
    """
    Отвечает *только* за преобразование непрерывных
    значений состояния (дистанция, угол) в дискретные "корзины" (bins)
    для использования в Q-таблице.
    """

    def __init__(self):
        """
        Инициализирует "сетку" для нарезки.
        Этот класс *запоминает* (хранит в 'self') 
        эти массивы-сетки.
        """
        
        # 1. Создаем "сетку" для дистанции.
        # np.linspace(start, stop, num) создает массив 'num' точек
        # между 'start' и 'stop'.
        # Пример для NUM_DISTANCE_BINS = 10:
        # [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        self.dist_bins = np.linspace(
            0, 
            config.MAX_STATE_DISTANCE, 
            config.NUM_DISTANCE_BINS
        )
        
        # 2. Создаем "сетку" для углов.
        # Пример для NUM_ANGLE_BINS = 10:
        # [-3.14, -2.48, -1.82, -1.17, -0.52, 0.52, 1.17, 1.82, 2.48, 3.14]
        self.angle_bins = np.linspace(
            -config.MAX_STATE_ANGLE, 
            config.MAX_STATE_ANGLE, 
            config.NUM_ANGLE_BINS
        )

    def discretize(self, continuous_state):
        """
        Преобразует непрерывное состояние в дискретный индекс-кортеж.
        
        Args:
            continuous_state (tuple): (distance, angle)
            
        Returns:
            tuple (int, int): (dist_bin_index, angle_bin_index)
        """
        # 1. Распаковываем непрерывные значения
        distance, angle = continuous_state
        
        # 2. Находим индекс "корзины" для дистанции
        # np.digitize(x, bins) - это функция NumPy, которая
        # спрашивает: "в какую ячейку 'bins' попадает значение 'x'?"
        # Пример: если dist_bins = [0.0, 0.5, 1.0, 1.5, ...]
        # и distance = 1.23, то np.digitize вернет 3
        # (т.к. 1.23 > 1.0 (индекс 2) и <= 1.5 (индекс 3))
        dist_bin = np.digitize(distance, self.dist_bins)
        
        # 3. Находим индекс "корзины" для угла
        angle_bin = np.digitize(angle, self.angle_bins)
        
        # 4. "Обрезаем" (Clamping) значения
        # np.digitize может вернуть значение 'NUM_BINS' (10),
        # если число больше, чем в последней ячейке.
        # А нам нужны индексы от 0 до 9 (для 10 ячеек).
        # Эта строка гарантирует, что мы не выйдем за пределы 
        # нашей Q-таблицы (от 0 до NUM_BINS-1).
        dist_bin = min(config.NUM_DISTANCE_BINS - 1, max(0, dist_bin - 1))
        angle_bin = min(config.NUM_ANGLE_BINS - 1, max(0, angle_bin - 1))

        # 5. Возвращаем кортеж с индексами
        # Например: (3, 2)
        # Это и есть наше ДИСКРЕТНОЕ СОСТОЯНИЕ (s).
        return (dist_bin, angle_bin)