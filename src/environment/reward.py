import config

class RewardCalculator:
    def calculate(self, old_dist, new_dist, is_turn, reached, action_id):
        # 1. Прогресс к цели (усиленный)
        dist_reward = (old_dist - new_dist) * 20.0
        
        # 2. Штрафы
        turn_pen = config.TURN_PENALTY_FACTOR if is_turn else 0.0
        idle_pen = config.IDLE_PENALTY if action_id == 4 else 0.0
        
        # 3. Бонус
        bonus = config.REACH_GOAL_BONUS if reached else 0.0
        
        return dist_reward - turn_pen - idle_pen + bonus