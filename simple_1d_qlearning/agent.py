# agent.py
import numpy as np
import time

class QLearningAgent:
    """
    Агент, использующий Q-Learning.
    Содержит Q-таблицу и всю логику обучения.
    """
    def __init__(self, num_states, num_actions, bins, 
                 alpha=0.2, gamma=0.99, epsilon=1.0, #Learning Rate #Discount Factor #Exploration Rate
                 epsilon_decay=0.99, min_epsilon=0.01): #Затуханный Epsilon
        self.num_states = num_states
        self.num_actions = num_actions
        self.bins = bins

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        self.q_table = np.zeros((num_states, num_actions))

    def discretize_state(self, continuous_state):
        return np.digitize(continuous_state, self.bins) - 1

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, new_state, done):
        old_value = self.q_table[state, action]
        
        if done:
            next_max = 0.0
        else:
            next_max = np.max(self.q_table[new_state, :])
        
        target_value = reward + self.gamma * next_max
        update_term = self.alpha * (target_value - old_value)
        self.q_table[state, action] = old_value + update_term

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes):
        print("Начинаем обучение...")
        rewards_history = []
        
        for episode in range(num_episodes):
            continuous_state = env.reset()
            state = self.discretize_state(continuous_state)
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                new_continuous_state, reward, done = env.step(action)
                new_state = self.discretize_state(new_continuous_state)
                self.update_q_table(state, action, reward, new_state, done)
                
                state = new_state
                total_reward += reward
            
            self.decay_epsilon()
            rewards_history.append(total_reward)
            
            if (episode + 1) % 50 == 0:
                print(f"Эпизод: {episode+1}/{num_episodes}, Суммарная награда: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        print("Обучение завершено.")
        return rewards_history

    def demonstrate(self, env):
        print("Запуск демонстрации обученного агента...")
        continuous_state = env.reset()
        state = self.discretize_state(continuous_state)
        
        original_epsilon = self.epsilon
        self.epsilon = 0.0 
        
        for _ in range(env.max_steps):
            action = self.choose_action(state)
            new_continuous_state, reward, done = env.step(action)
            state = self.discretize_state(new_continuous_state)
            
            time.sleep(5.0/60.0) 
            
            if done:
                break
        
        self.epsilon = original_epsilon
        print("Демонстрация завершена.")