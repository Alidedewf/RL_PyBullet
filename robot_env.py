import pybullet as p
import pybullet_data
import numpy as np
import time
import config  # Наш файл с настройками

class RobotEnv:
    """
    Кастомная среда для обучения робота с использованием PyBullet.
    """

    def __init__(self):
        # 1. Инициализация PyBullet
        if config.GUI_MODE:
            # Подключаемся с графическим интерфейсом
            self.client_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            # Устанавливаем "камеру" в удобную позицию
            p.resetDebugVisualizerCamera(cameraDistance=5, 
                                         cameraYaw=0, 
                                         cameraPitch=-40, 
                                         cameraTargetPosition=[0, 0, 0])
        else:
            # Подключаемся без GUI для быстрого обучения
            self.client_id = p.connect(p.DIRECT)

        # Настраиваем симулятор
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0) # Мы будем сами вызывать stepSimulation
        p.setTimeStep(config.SIM_TIMESTEP)

        # 2. Загрузка сцены и объектов
        # Загружаем пол
        p.loadURDF(config.PLANE_URDF_PATH)

        # Загружаем робота
        # Начальная позиция робота [x, y, z]
        start_pos = [0, 0, 0.1] 
        # Начальная ориентация (в кватернионах)
        start_orientation = p.getQuaternionFromEuler([0, 0, 0]) 
        self.robot_id = p.loadURDF(config.ROBOT_URDF_PATH, start_pos, start_orientation)

        # Загружаем цель (визуальный объект - сфера)
        # Создаем форму (визуальную)
        target_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=[1, 0, 0, 1] # Красный цвет
        )
        
        # Создаем "тело" объекта (без физики, -1)
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual_shape,
            basePosition=[0, 0, 0] # Позиция будет задана в reset()
        )
        
        # 3. Настройка "корзин" для дискретизации
        # Линейное пространство для дистанции
        self.dist_bins = np.linspace(0, 
                                     config.MAX_STATE_DISTANCE, 
                                     config.NUM_DISTANCE_BINS)
        # Линейное пространство для углов (от -PI до +PI)
        self.angle_bins = np.linspace(-config.MAX_STATE_ANGLE, 
                                       config.MAX_STATE_ANGLE, 
                                       config.NUM_ANGLE_BINS)

        # Внутренние переменные
        self.episode_steps = 0
        self.old_distance = 0.0


    def reset(self):
        """
        Перезапускает эпизод.
        1. Перемещает робота в (0, 0).
        2. Спавнит цель в случайной позиции.
        3. Возвращает первое дискретное состояние.
        """
        self.episode_steps = 0

        # Сброс позиции и скорости робота
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.1], [0, 0, 0, 1])
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])

        # Спавн цели в новой случайной точке
        self.target_pos_2d = self._spawn_target()
        
        # Получаем первое состояние
        # 'continuous_state' - это (дистанция, угол)
        continuous_state, _ = self._get_state()
        
        # Запоминаем дистанцию для расчета награды на следующем шаге
        self.old_distance = continuous_state[0] 
        
        # Возвращаем дискретное состояние (bin_dist, bin_angle)
        return self._discretize_state(continuous_state)

    def _spawn_target(self):
        """ Вспомогательная функция для спавна цели. """
        # Случайная позиция в пределах радиуса
        radius = np.random.uniform(1.0, config.MAX_TARGET_SPAWN_RADIUS) # Не спавним слишком близко
        angle = np.random.uniform(-np.pi, np.pi)
        
        target_x = radius * np.cos(angle)
        target_y = radius * np.sin(angle)
        
        # Перемещаем объект цели (на высоте 0.1)
        p.resetBasePositionAndOrientation(self.target_id, [target_x, target_y, 0.1], [0, 0, 0, 1])
        
        return np.array([target_x, target_y])


    def _get_state(self):
        """
        Вычисляет непрерывное состояние: (дистанция_до_цели, угол_до_цели).
        Также возвращает текущий 'yaw' (угол поворота) робота.
        """
        # 1. Позиция и ориентация робота
        pos, orient_q = p.getBasePositionAndOrientation(self.robot_id)
        # Конвертируем кватернион ориентации в углы Эйлера
        orient_euler = p.getEulerFromQuaternion(orient_q)
        
        robot_pos_2d = np.array([pos[0], pos[1]])
        robot_yaw = orient_euler[2] # Угол "рыскания" (вокруг оси Z)

        # 2. Векторы
        vec_to_target = self.target_pos_2d - robot_pos_2d

        # 3. Дистанция (непрерывное значение)
        # Мы используем 2D-дистанцию (x, y), игнорируя высоту
        distance = np.linalg.norm(vec_to_target)

        # 4. Угол до цели (непрерывное значение)
        # Угол вектора на цель в мировой системе координат
        angle_to_target_global = np.arctan2(vec_to_target[1], vec_to_target[0])
        
        # Угол до цели ОТНОСИТЕЛЬНО "носа" робота
        # Это то, что "видит" робот
        angle_relative = angle_to_target_global - robot_yaw

        # Нормализуем угол в диапазон [-pi, pi]
        # (Если робот смотрит на 170 град, а цель на -170, 
        # угол должен быть 20, а не 340)
        angle_relative = np.arctan2(np.sin(angle_relative), np.cos(angle_relative))
        
        continuous_state = (distance, angle_relative)
        return continuous_state, robot_yaw


    def _discretize_state(self, continuous_state):
        """
        Преобразует непрерывное состояние (дистанция, угол) 
        в дискретные "корзины" (индексы).
        """
        distance, angle = continuous_state
        
        # np.digitize находит, в какую "корзину" попадает значение
        # Мы "обрезаем" значения, которые выходят за рамки
        dist_bin = np.digitize(distance, self.dist_bins)
        angle_bin = np.digitize(angle, self.angle_bins)
        
        # Убедимся, что индексы не выходят за пределы
        dist_bin = min(config.NUM_DISTANCE_BINS - 1, max(0, dist_bin - 1))
        angle_bin = min(config.NUM_ANGLE_BINS - 1, max(0, angle_bin - 1))

        return (dist_bin, angle_bin) # Это будет ключ для Q-таблицы


    def step(self, action_id):
        """
        Выполняет один шаг симуляции.
        Принимает: action_id (0, 1, 2, 3, 4)
        Возвращает: (новое_дискретное_состояние, награда, готово)
        """
        
        # 1. Применяем действие
        # Получаем скорости (v_left, v_right) из нашего конфига
        v_left, v_right = config.ACTIONS_MAP[action_id]
        
        # Отправляем команду моторам
        p.setJointMotorControl2(
            self.robot_id,
            config.LEFT_WHEEL_JOINT_INDEX,
            p.VELOCITY_CONTROL,
            targetVelocity=v_left,
            force=config.MAX_WHEEL_FORCE
        )
        p.setJointMotorControl2(
            self.robot_id,
            config.RIGHT_WHEEL_JOINT_INDEX,
            p.VELOCITY_CONTROL,
            targetVelocity=v_right,
            force=config.MAX_WHEEL_FORCE
        )

        # 2. Выполняем шаг симуляции
        p.stepSimulation()
        if config.GUI_MODE:
            # Маленькая пауза для "реального времени" в GUI
            time.sleep(config.SIM_TIMESTEP) 

        self.episode_steps += 1

        # 3. Получаем новое состояние
        (new_distance, new_angle), robot_yaw = self._get_state()
        new_discrete_state = self._discretize_state((new_distance, new_angle))

        # 4. Рассчитываем награду
        reward, done = self._calculate_reward(new_distance, action_id)

        # 5. Обновляем 'old_distance' для следующего шага
        self.old_distance = new_distance
        
        # 6. Проверяем, не закончился ли эпизод по времени
        if self.episode_steps >= config.MAX_EPISODE_STEPS:
            done = True

        return new_discrete_state, reward, done

    def _calculate_reward(self, new_distance, action_id):
        """
        Расчет награды по формуле из задания.
        R = -(distance_new - distance_old) - 0.01 * turn_penalty + Bonus
        """
        done = False
        
        # 1. Награда за приближение
        # (Если new < old, награда будет positive)
        distance_reward = self.old_distance - new_distance

        # 2. Штраф за поворот (избыточные маневры)
        turn_penalty = 0.0
        if action_id == 2 or action_id == 3: # 2: Влево, 3: Вправо
            turn_penalty = config.TURN_PENALTY_FACTOR

        # 3. Бонус за достижение цели
        goal_bonus = 0.0
        if new_distance < config.DISTANCE_THRESHOLD:
            goal_bonus = config.REACH_GOAL_BONUS
            done = True # Эпизод успешно завершен
            
        # Итоговая награда
        total_reward = distance_reward - turn_penalty + goal_bonus
        
        return total_reward, done

    def close(self):
        """Отключается от симулятора."""
        p.disconnect(self.client_id)