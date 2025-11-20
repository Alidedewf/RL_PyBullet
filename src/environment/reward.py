import config # Нам нужны константы наград из конфига

class RewardCalculator:
    """
    Отвечает *только* за вычисление награды.
    Это "без-состояния" (stateless) калькулятор.
    Он реализует формулу:
    R = (dist_old - dist_new)*100 - turn_penalty + Bonus
    """

    def __init__(self):
        """
        Этот класс - чистый калькулятор, ему не нужно 
        ничего хранить или инициализировать.
        """
        pass

    def calculate_reward(self, old_distance, new_distance, is_turn_action, goal_reached):
        
        """
        Рассчитывает итоговую награду за один шаг.
        
        Args:
            old_distance (float): Дистанция до цели на *предыдущем* шаге.
            new_distance (float): Дистанция до цели на *текущем* шаге.
            is_turn_action (bool): True, если было совершено действие "поворот".
            goal_reached (bool): True, если цель была достигнута на этом шаге.
            
        Returns:
            float: Итоговая, суммарная награда.
        """

        # ================================================================
        # --- 1. Награда за приближение (Reward Shaping) ---
        # 
        # !!! ЭТО НАШЕ ГЛАВНОЕ ИСПРАВЛЕНИЕ !!!
        # Мы умножаем разницу на 100.
        # Раньше: (3.1 - 3.0) = +0.1 (слишком "тихий" сигнал)
        # Сейчас: (3.1 - 3.0) * 100 = +10.0 (мощный, "громкий" сигнал)
        # ================================================================
        distance_reward = (old_distance - new_distance) * 100.0

        # --- 2. Штраф за поворот (Turn Penalty) ---
        # Мы не хотим, чтобы робот "танцевал" на месте.
        # Мы штрафуем его на маленькое значение, если он 
        # выбрал действие "поворот".
        turn_penalty = 0.0
        if is_turn_action:
            # Берем константу штрафа из нашего config.py
            turn_penalty = config.TURN_PENALTY_FACTOR 

        # --- 3. Бонус за достижение цели (Goal Bonus) ---
        # Это "джекпот", который робот получает, когда
        # подъезжает достаточно близко.
        goal_bonus = 0.0
        if goal_reached:
            # Берем константу бонуса из config.py
            goal_bonus = config.REACH_GOAL_BONUS

        step_penalty = 0.0
        if not goal_reached:
            step_penalty = config.STEP_PENALTY
            
        # --- 4. Итоговая награда ---
        # Складываем все части вместе.
        total_reward = distance_reward - turn_penalty + goal_bonus - step_penalty
                
        return total_reward