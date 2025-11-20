import numpy as np
import time
import config  # Наш файл с настройками
import sys     
import os

# Добавляем папку 'src' в путь поиска
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.base_env import RobotEnv
from agent.q_agent import QLearningAgent
from utils import plot_rewards

def main():
    # =================================================================
    # ЭТАП 1: ОБУЧЕНИЕ (БЫСТРО, БЕЗ ГРАФИКИ) ⚡️
    # =================================================================
    
    # Принудительно отключаем GUI для скорости
    config.GUI_MODE = False 
    print(f"--- ЭТАП 1: ОБУЧЕНИЕ (GUI выключен) ---")
    print(f"Цель: {config.NUM_EPISODES} эпизодов. Пожалуйста, подождите...")

    # Создаем среду для обучения
    env = RobotEnv()
    agent = QLearningAgent()
    episode_rewards_list = []
    
    start_time = time.time()

    try:
        for episode in range(config.NUM_EPISODES):
            state = env.reset()
            total_episode_reward = 0
            done = False
            
            for step in range(config.MAX_EPISODE_STEPS):
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state)
                
                state = next_state
                total_episode_reward += reward
                
                if done:
                    break
            
            episode_rewards_list.append(total_episode_reward)
            agent.decay_epsilon()

            # Логирование каждые 100 эпизодов
            if (episode + 1) % 100 == 0:
                avg = np.mean(episode_rewards_list[-100:])
                print(f"\rЭпизод: {episode + 1} | Epsilon: {agent.epsilon:.2f} | Награда: {avg:.2f}", end="")

        print("\n--- Обучение завершено! ---")
        print(f"Время обучения: {(time.time() - start_time) / 60:.2f} мин.")
        
        # Закрываем "слепую" среду
        env.close()
        
        # Строим график (сохранится в файл)
        plot_rewards(episode_rewards_list)

    except KeyboardInterrupt:
        print("\nОбучение прервано.")
        env.close()
        return

    # =================================================================
    # ЭТАП 2: ДЕМОНСТРАЦИЯ (КРАСИВО, С ГРАФИКОЙ) 📺
    # =================================================================
    
    print(f"\n--- ЭТАП 2: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТА ---")
    print(f"Запускаем умного робота...")
    
    # Принудительно ВКЛЮЧАЕМ GUI
    config.GUI_MODE = True
    
    # Отключаем случайность (Робот использует только знания)
    agent.epsilon = 0.0 
    
    # Создаем НОВУЮ среду, уже с графикой
    # (PyBullet требует пересоздания для смены режима)
    env_gui = RobotEnv()
    
    try:
        # Запустим 10 показательных выступлений
        for demo_ep in range(10):
            state = env_gui.reset()
            done = False
            total_reward = 0
            print(f"Демонстрация {demo_ep + 1}/10...", end="")
            
            for step in range(config.MAX_EPISODE_STEPS):
                # Агент выбирает ЛУЧШЕЕ действие (epsilon=0)
                action = agent.choose_action(state)
                
                # ВНИМАНИЕ: Тут мы не вызываем agent.learn(), 
                # потому что на экзамене не учатся :)
                next_state, reward, done = env_gui.step(action)
                
                state = next_state
                total_reward += reward
                
                if done:
                    print(f" УСПЕХ! 🎯 (Награда: {total_reward:.2f})")
                    time.sleep(1) # Пауза, чтобы порадоваться победе
                    break
            
            if not done:
                print(f" Время вышло. (Награда: {total_reward:.2f})")
                
    except KeyboardInterrupt:
        print("\nПросмотр завершен.")
    finally:
        env_gui.close()
        print("Программа завершена.")

if __name__ == "__main__":
    main()