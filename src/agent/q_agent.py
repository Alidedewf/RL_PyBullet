import numpy as np
import config # Наш файл с настройками

class QLearningAgent:
    """
    Агент, реализующий Q-Learning с Epsilon-Greedy стратегией.
    """
    
    def __init__(self):
        # 1. Инициализация Q-таблицы
        # Размер таблицы: (корзины_дистанции, корзины_угла, кол-во_действий)
        # Например, (10, 10, 5)
        q_table_shape = (
            config.NUM_DISTANCE_BINS,
            config.NUM_ANGLE_BINS,
            config.NUM_ACTIONS
        )
        
        # Инициализируем таблицу нулями.
        self.q_table = np.zeros(q_table_shape)

        # 2. Параметры обучения из конфига
        self.alpha = config.ALPHA 
        self.gamma = config.GAMMA 
        
        # 3. Параметры Epsilon-Greedy
        self.epsilon = config.EPSILON_START
        
        self.epsilon_decay_value = (config.EPSILON_START - config.EPSILON_END) / \
                                     config.EPSILON_DECAY_DURATION
                                     
    def choose_action(self, state):
        """
        Выбирает действие (0-4) на основе текущего состояния.
        'state' - это кортеж (dist_bin, angle_bin)
        """
        
        random_num = np.random.uniform(0, 1)

        # 1. Случай (Исследование / Exploration)
        if random_num < self.epsilon:
            action = np.random.choice(config.NUM_ACTIONS)
        
        # 2. Случай (Эксплуатация / Exploitation)
        else:
            state_q_values = self.q_table[state]
            action = np.argmax(state_q_values)
            
        return action

    def learn(self, state, action, reward, next_state):
        """
        Обновляет Q-таблицу на основе полученного опыта.
        Q[s,a] = Q[s,a] + α * [r + γ * max_a'(Q[s',a']) - Q[s,a]]
        """
        
        # Q[s, a]
        old_q_value = self.q_table[state][action]
        
        # max_a'(Q[s', a'])
        max_q_next = np.max(self.q_table[next_state])

        # r + γ * max_a'(Q[s', a'])
        td_target = reward + self.gamma * max_q_next

        # [r + γ * max_a'(Q[s', a']) - Q[s, a]]
        td_error = td_target - old_q_value

        # Q[s, a] = Q[s, a] + α * (TD Error)
        self.q_table[state][action] = old_q_value + self.alpha * td_error

    def decay_epsilon(self):
        """
        Уменьшает эпсилон после каждого эпизода.
        """
        if self.epsilon > config.EPSILON_END:
            self.epsilon -= self.epsilon_decay_value
        else:
            self.epsilon = config.EPSILON_END