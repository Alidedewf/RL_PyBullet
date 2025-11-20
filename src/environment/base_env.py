import config

# --- Импортируем всех наших "помощников" ---
from src.environment.simulation import SimulationManager
from src.environment.robot import Robot
from src.environment.task import Task
from src.environment.state import StateCalculator
from src.environment.reward import RewardCalculator
from src.environment.discretizer import StateDiscretizer

class RobotEnv:
    """
    "Дирижер" - главный класс Среды (Environment).
    Он связывает все компоненты (симуляцию, робота, задачу,
    калькуляторы) вместе, чтобы предоставить стандартный
    интерфейс (reset, step) для нашего Агента.
    """

    def __init__(self):
        """
        Инициализирует и "собирает" все части среды.
        """
        # 1. Запускаем симуляцию
        self.sim_manager = SimulationManager()
        # Получаем "ключ" к симуляции
        client_id = self.sim_manager.get_client_id()
        
        # 2. Создаем "физические" объекты, передавая им "ключ"
        self.robot = Robot(client_id)
        self.task = Task(client_id)
        
        # 3. Создаем "калькуляторы"
        self.state_calc = StateCalculator()
        self.reward_calc = RewardCalculator()
        self.discretizer = StateDiscretizer()
        
        # 4. Внутренние переменные состояния
        self.episode_steps = 0
        # Нам *необходимо* помнить старую дистанцию между шагами
        # для расчета награды.
        self.old_distance = 0.0

    def reset(self):
        """
        Перезапускает эпизод и возвращает первое дискретное состояние.
        """
        # 1. Сбрасываем счетчик шагов
        self.episode_steps = 0
        
        # 2. Говорим объектам сбросить свое состояние
        self.robot.reset()
        target_pos_2d = self.task.reset() # Запоминаем новую позицию цели
        for _ in range(50):
            self.sim_manager.step()
        
        # 3. Получаем "сырые" данные от робота
        robot_pos_2d, robot_yaw = self.robot.get_observation()
        
        # 4. Вычисляем первое непрерывное состояние
        # Делегируем это калькулятору
        continuous_state = self.state_calc.calculate_state(
            robot_pos_2d, robot_yaw, target_pos_2d
        )
        
        # 5. Сохраняем первую дистанцию для расчета награды на 1-м шаге
        distance, angle = continuous_state
        self.old_distance = distance
        
        # 6. Превращаем (дистанцию, угол) в (индекс_1, индекс_2)
        # Делегируем это дискретизатору
        discrete_state = self.discretizer.discretize(continuous_state)
        
        # 7. Возвращаем состояние для Агента
        return discrete_state

    def step(self, action_id):
        # 1. Применяем действие
        self.robot.apply_action(action_id)
        
        # 2. Физика
        self.sim_manager.step()
        
        # 3. Получаем данные
        robot_pos_2d, robot_yaw = self.robot.get_observation()
        target_pos_2d = self.task.get_target_position()
        
        # 4. Считаем состояние (дистанция и УГОЛ)
        new_continuous_state = self.state_calc.calculate_state(
            robot_pos_2d, robot_yaw, target_pos_2d
        )
        new_distance, new_angle = new_continuous_state # <--- Берем угол!
        
        # 5. Проверка цели
        goal_reached = self.task.check_goal_reached(robot_pos_2d)
        
        # 6. Награда
        is_turn = (action_id == 2 or action_id == 3)
        
        # !!! ВАЖНО: Передаем new_angle в калькулятор !!!
        # (Если у тебя в reward.py метод принимает 5 аргументов)
        try:
            reward = self.reward_calc.calculate_reward(
                self.old_distance, 
                new_distance, 
                new_angle,      # <-- Добавь этот аргумент
                is_turn, 
                goal_reached
            )
        except TypeError:
            # Если ты НЕ обновлял reward.py, оставь старый вызов:
            reward = self.reward_calc.calculate_reward(
                self.old_distance, new_distance, is_turn, goal_reached
            )

        # 7. Обновляем память
        self.old_distance = new_distance
        self.episode_steps += 1
        
        # 8. Дискретизация
        new_discrete_state = self.discretizer.discretize(new_continuous_state)
        
        # 9. Done
        done = False
        if goal_reached:
            done = True
        elif self.episode_steps >= config.MAX_EPISODE_STEPS:
            done = True
            
        return new_discrete_state, reward, done
        """
        Выполняет один шаг симуляции.
        Принимает: ID действия (0-4)
        Возвращает: (новое_дискретное_состояние, награда, готово)
        """
        
        # 1. Применяем действие к роботу
        self.robot.apply_action(action_id)
        
        # 2. Выполняем шаг физики
        self.sim_manager.step()
        
        # 3. Получаем новые "сырые" данные
        robot_pos_2d, robot_yaw = self.robot.get_observation()
        target_pos_2d = self.task.get_target_position()
        
        # 4. Вычисляем новое непрерывное состояние
        new_continuous_state = self.state_calc.calculate_state(
            robot_pos_2d, robot_yaw, target_pos_2d
        )
        new_distance, new_angle = new_continuous_state
        
        # 5. Проверяем, достигнута ли цель
        # Делегируем это "Задаче"
        goal_reached = self.task.check_goal_reached(robot_pos_2d)
        
        # 6. Рассчитываем награду
        # Делегируем это "Калькулятору Наград"
        is_turn = (action_id == 2 or action_id == 3) # Это поворот?
        reward = self.reward_calc.calculate_reward(
            self.old_distance, 
            new_distance, 
            is_turn, 
            goal_reached
        )
        
        # 7. Обновляем "внутреннюю память"
        self.old_distance = new_distance
        self.episode_steps += 1
        
        # 8. Превращаем новое состояние в дискретное
        new_discrete_state = self.discretizer.discretize(new_continuous_state)
        
        # 9. Проверяем, не закончился ли эпизод
        done = False
        if goal_reached:
            done = True
        elif self.episode_steps >= config.MAX_EPISODE_STEPS:
            done = True
            
        # 10. Возвращаем результат для Агента
        return new_discrete_state, reward, done

    def close(self):
        """
        Корректно отключается от симуляции.
        """
        # Делегируем это Менеджеру Симуляции
        self.sim_manager.close()