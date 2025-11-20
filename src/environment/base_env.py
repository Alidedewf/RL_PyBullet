import config

# --- Импорты ---
from src.environment.simulation import SimulationManager
from src.environment.robot import Robot
from src.environment.task import Task
from src.environment.state import StateCalculator
from src.environment.reward import RewardCalculator
from src.environment.discretizer import StateDiscretizer

class RobotEnv:
    """
    Главный класс среды с Action Repeat для плавности.
    """

    def __init__(self):
        self.sim_manager = SimulationManager()
        client_id = self.sim_manager.get_client_id()
        
        self.robot = Robot(client_id)
        self.task = Task(client_id)
        
        self.state_calc = StateCalculator()
        self.reward_calc = RewardCalculator()
        self.discretizer = StateDiscretizer()
        
        self.episode_steps = 0
        self.old_distance = 0.0

    def reset(self):
        self.episode_steps = 0
        
        self.robot.reset()
        target_pos_2d = self.task.reset()
        
        # Стабилизация: даем роботу упасть и встать на колеса
        for _ in range(50):
            self.sim_manager.step()
        
        robot_pos_2d, robot_yaw = self.robot.get_observation()
        
        continuous_state = self.state_calc.calculate_state(
            robot_pos_2d, robot_yaw, target_pos_2d
        )
        
        self.old_distance = continuous_state[0]
        
        return self.discretizer.discretize(continuous_state)

    def step(self, action_id):
        """
        Выполняет действие с повторением (Action Repeat).
        """
        
        # 1. Применяем действие к моторам
        self.robot.apply_action(action_id)
        
        # 2. !!! ACTION REPEAT (Плавность) !!!
        # Мы прокручиваем физику N раз, пока робот выполняет ОДНО решение.
        # Это убирает дерганье.
        total_reward = 0
        goal_reached = False
        
        for _ in range(config.ACTION_REPEAT):
            self.sim_manager.step()
            
            # Проверяем цель внутри микро-шагов, чтобы не проскочить
            robot_pos_2d, _ = self.robot.get_observation()
            if self.task.check_goal_reached(robot_pos_2d):
                goal_reached = True
                break # Выходим из цикла повторений, если попали
        
        # 3. Получаем финальное состояние после движения
        robot_pos_2d, robot_yaw = self.robot.get_observation()
        target_pos_2d = self.task.get_target_position()
        
        new_continuous_state = self.state_calc.calculate_state(
            robot_pos_2d, robot_yaw, target_pos_2d
        )
        new_distance, new_angle = new_continuous_state
        
        # 4. Рассчитываем награду
        is_turn = (action_id == 2 or action_id == 3)
        
        # Передаем action_id, чтобы штрафовать за стоянку (ID 4)
        reward = self.reward_calc.calculate_reward(
            self.old_distance, 
            new_distance, 
            is_turn, 
            goal_reached,
            action_id 
        )
        
        self.old_distance = new_distance
        self.episode_steps += 1
        
        new_discrete_state = self.discretizer.discretize(new_continuous_state)
        
        # 5. Условия завершения
        done = False
        if goal_reached:
            done = True
        elif self.episode_steps >= config.MAX_EPISODE_STEPS:
            done = True
            
        return new_discrete_state, reward, done

    def close(self):
        self.sim_manager.close()