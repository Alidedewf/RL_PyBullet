import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(episode_rewards):
    window_size = 100
    plt.figure(figsize=(12, 6))
    plt.title(f"Награды за эпизод (Скользящее среднее окно {window_size})")
    plt.xlabel("Эпизод")
    plt.ylabel("Суммарная награда")
    
    plt.plot(episode_rewards, label="Награда (сырая)", color='cyan', alpha=0.3)
    
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(episode_rewards)), moving_avg, label="Среднее", color='red', linewidth=2)

    plt.legend()
    plt.grid(True)
    plt.savefig("training_rewards.png")
    print(f"График сохранен в 'training_rewards.png'")