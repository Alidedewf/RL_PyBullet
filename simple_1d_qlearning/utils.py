# utils.py
import matplotlib.pyplot as plt

def plot_rewards(rewards_history):
    """
    Отдельный метод для отрисовки графиков.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title("Динамика суммарной награды по эпизодам")
    plt.xlabel("Эпизод")
    plt.ylabel("Суммарная награда")
    plt.grid(True)
    plt.show()