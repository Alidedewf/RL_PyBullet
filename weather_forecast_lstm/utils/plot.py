import matplotlib.pyplot as plt
import numpy as np

def plot_forecast(true_vals, predicted_vals, alt_preds=None):
    days = range(len(true_vals))

    plt.figure(figsize=(7, 5))

    # Настройка стиля
    plt.style.use("seaborn-v0_8-darkgrid")  # стиль фона

    # Реальные значения — фиолетовая линия с кружками
    plt.plot(days, true_vals, label="📘 Реальные значения",
             color="#6A5ACD", marker="o", linestyle="-", linewidth=2)

    # Основной прогноз — тёплый оранжевый крестиками
    plt.plot(days, predicted_vals, label="🟠 Прогноз (основной)",
             color="#FFA500", marker="x", linestyle="--", linewidth=2)

    # Альтернативный прогноз (если передан)
    if alt_preds is not None:
        plt.plot(days, alt_preds, label="🔵 Альтернативный прогноз",
                 color="#00BFFF", marker="s", linestyle=":", linewidth=2)

    # Заливка между предсказаниями и фактами
    plt.fill_between(days, true_vals, predicted_vals,
                     color="#FFE4B5", alpha=0.3, label="Отклонение прогноза")

    # Заголовок и подписи
    plt.title("🌤️ Сравнение прогноза температуры на 30 дней вперёд", fontsize=16)
    plt.xlabel("День", fontsize=12)
    plt.ylabel("Температура (°C)", fontsize=12)
    plt.xticks(days)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()