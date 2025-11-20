import config 

class RewardCalculator:
    def __init__(self):
        pass

    def calculate_reward(self, old_distance, new_distance, is_turn_action, goal_reached, action_id):
        
        # 1. Награда за приближение (масштаб уменьшен до 20, чтобы не перебить бонус)
        # Если проедет 2 метра: 2.0 * 20 = 40 очков.
        distance_reward = (old_distance - new_distance) * 20.0

        # 2. Штраф за поворот
        turn_penalty = 0.0
        if is_turn_action:
            turn_penalty = config.TURN_PENALTY_FACTOR 

        # 3. Штраф за простой (Action ID 4 - Стоять)
        idle_penalty = 0.0
        if action_id == 4:
            idle_penalty = config.IDLE_PENALTY

        # 4. БОНУС (Главная цель)
        goal_bonus = 0.0
        if goal_reached:
            goal_bonus = config.REACH_GOAL_BONUS # Это 100
            
        # ИТОГ
        # Если доехал: 100 (бонус) + ~40 (путь) - штрафы = ~130-140 макс.
        # Если стоим: -0.5 каждый шаг.
        total_reward = distance_reward - turn_penalty - idle_penalty + goal_bonus
                
        return total_reward