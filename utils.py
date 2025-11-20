import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(episode_rewards):
    """
    Строит и сохраняет график суммарных наград за эпизоды.
    
    На графике отображаются:
    1. Сырые данные (прозрачные) - награда за каждый отдельный эпизод.
    2. Скользящее среднее (жирное) - для отслеживания тренда.
    """
    
    # 1. Настройка окна в 100 эпизодов
    # Это соответствует критерию "средняя награда за последние 100 эпизодов"
    window_size = 100
    
    plt.figure(figsize=(12, 6))
    plt.title(f"Награды за эпизод (Скользящее среднее с окном {window_size})")
    plt.xlabel("Эпизод")
    plt.ylabel("Суммарная награда")

    # 2. Отрисовка сырых данных (награда за каждый эпизод)
    # Мы делаем их полупрозрачными (alpha=0.3), т.к. они очень "шумные"
    plt.plot(episode_rewards, label="Награда за эпизод (сырые)", 
             color='cyan', alpha=0.3)

    # 3. Расчет и отрисовка скользящего среднего
    # Убедимся, что у нас достаточно данных для расчета
    if len(episode_rewards) >= window_size:
        
        # np.convolve - это быстрый и стандартный способ расчета 
        # скользящего среднего в numpy.
        # Мы "сворачиваем" массив наград с массивом из 100 
        # одинаковых весов (1/100).
        moving_avg = np.convolve(episode_rewards, 
                                 np.ones(window_size) / window_size, 
                                 mode='valid')
                                 
        # Важно: ось X для скользящего среднего сдвинута.
        # Первое значение moving_avg - это среднее эпизодов [0...99],
        # поэтому мы должны "поставить" эту точку на 99-й эпизод.
        x_axis = np.arange(window_size - 1, len(episode_rewards))
        
        plt.plot(x_axis, moving_avg, 
                 label=f"Скользящее среднее (окно {window_size})", 
                 color='red', linewidth=2)

    plt.legend()
    plt.grid(True)
    
    # Сохраняем график в файл
    plt.savefig("training_rewards.png")
    print(f"\nГрафик наград сохранен в 'training_rewards.png'")
    
    # Показываем график пользователю
    plt.show()