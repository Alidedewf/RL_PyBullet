# main.py
import numpy as np
from environment import SimpleEnv
from agent import QLearningAgent
from utils import plot_rewards

if __name__ == "__main__":
    
    NUM_EPISODES = 1000
    
    pos_bins = np.linspace(-5.0, 10.0, 40) 
    num_states = len(pos_bins) + 1
    num_actions = 2 

    agent = QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        bins=pos_bins
    )
    
    train_env = SimpleEnv(render=False)
    
    history = agent.train(train_env, NUM_EPISODES)
    
    train_env.close()

    plot_rewards(history)

    demo_env = SimpleEnv(render=True)
    
    agent.demonstrate(demo_env)
    
    demo_env.close()