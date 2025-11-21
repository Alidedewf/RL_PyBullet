import config
from .simulation import SimulationManager
from .robot import Robot
from .task import Task
from .state import StateCalculator
from .reward import RewardCalculator
from .discretizer import StateDiscretizer

class RobotEnv:
    def __init__(self):
        self.sim = SimulationManager()
        cid = self.sim.get_client_id()
        
        self.robot = Robot(cid)
        self.task = Task(cid)
        self.state_calc = StateCalculator()
        self.reward_calc = RewardCalculator()
        self.discretizer = StateDiscretizer()
        self.old_dist = 0.0
        self.steps = 0

    def reset(self):
        self.steps = 0
        self.robot.reset()
        target_pos = self.task.reset()
        
        # Стабилизация Husky (чтобы не прыгал)
        for _ in range(50): self.sim.step()
        
        robot_pos, robot_yaw = self.robot.get_observation()
        cont_state = self.state_calc.calculate_state(robot_pos, robot_yaw, target_pos)
        self.old_dist = cont_state[0]
        return self.discretizer.discretize(cont_state)

    def step(self, action_id):
        self.robot.apply_action(action_id)
        
        # Action Repeat (плавность)
        reached = False
        for _ in range(config.ACTION_REPEAT):
            self.sim.step()
            pos, _ = self.robot.get_observation()
            if self.task.check_goal_reached(pos):
                reached = True
                break
        
        # Получение состояния
        robot_pos, robot_yaw = self.robot.get_observation()
        target_pos = self.task.get_target_position()
        cont_state = self.state_calc.calculate_state(robot_pos, robot_yaw, target_pos)
        dist, angle = cont_state
        
        # Расчет награды
        is_turn = (action_id == 2 or action_id == 3)
        reward = self.reward_calc.calculate(self.old_dist, dist, is_turn, reached, action_id)
        
        self.old_dist = dist
        self.steps += 1
        
        done = reached or (self.steps >= config.MAX_EPISODE_STEPS)
        return self.discretizer.discretize(cont_state), reward, done

    def close(self):
        self.sim.close()