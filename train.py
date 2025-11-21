import numpy as np
import time
import config
import sys
import os

# Добавляем src в путь, чтобы Python видел модули
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.base_env import RobotEnv
from agent.q_agent import QLearningAgent
from utils import plot_rewards

def main():
    # --- ФАЗА 1: ОБУЧЕНИЕ (БЫСТРО) ---
    config.GUI_MODE = False 
    print(f"--- НАЧАЛО ОБУЧЕНИЯ ({config.NUM_EPISODES} эпизодов, GUI выкл) ---")

    env = RobotEnv()
    agent = QLearningAgent()
    rewards_history = []
    
    start_time = time.time()

    try:
        for episode in range(config.NUM_EPISODES):
            state = env.reset()
            total_reward = 0
            done = False
            
            for step in range(config.MAX_EPISODE_STEPS):
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
                if done: break
            
            rewards_history.append(total_reward)
            agent.decay_epsilon()

            if (episode + 1) % 100 == 0:
                avg = np.mean(rewards_history[-100:])
                print(f"\rЭпизод: {episode + 1} | Epsilon: {agent.epsilon:.2f} | Средняя награда: {avg:.2f}", end="")

    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем.")
    finally:
        env.close()
        print(f"\nВремя обучения: {(time.time() - start_time) / 60:.2f} мин.")
        plot_rewards(rewards_history)

    # --- ФАЗА 2: ДЕМОНСТРАЦИЯ (КРАСИВО) ---
    print(f"\n\n--- ЗАПУСК ДЕМОНСТРАЦИИ (GUI вкл) ---")
    config.GUI_MODE = True
    agent.epsilon = 0.0 # Отключаем случайность, используем опыт
    
    env_gui = RobotEnv()
    
    try:
        for i in range(10):
            state = env_gui.reset()
            done = False
            score = 0
            print(f"Тест {i+1}/10...", end="")
            
            for _ in range(config.MAX_EPISODE_STEPS):
                action = agent.choose_action(state)
                next_state, reward, done = env_gui.step(action)
                state = next_state
                score += reward
                if done:
                    print(f" УСПЕХ! 🎯 (Награда: {score:.2f})")
                    time.sleep(1)
                    break
            if not done:
                print(f" Не доехал. (Награда: {score:.2f})")
                
    except KeyboardInterrupt:
        pass
    finally:
        env_gui.close()
        print("Готово.")

if __name__ == "__main__":
    main()