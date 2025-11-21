import numpy as np
import config

class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((config.NUM_DISTANCE_BINS, config.NUM_ANGLE_BINS, config.NUM_ACTIONS))
        self.alpha = config.ALPHA
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.decay = (config.EPSILON_START - config.EPSILON_END) / config.EPSILON_DECAY_DURATION

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(config.NUM_ACTIONS)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        old_val = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_val = old_val + self.alpha * (reward + self.gamma * next_max - old_val)
        self.q_table[state][action] = new_val

    def decay_epsilon(self):
        if self.epsilon > config.EPSILON_END:
            self.epsilon -= self.decay