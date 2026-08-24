# weather_forecast_lstm

Прогноз максимальной суточной температуры в Астане на LSTM, по 15 годам исторических данных.

| Модуль | Роль |
|---|---|
| `data/loader.py` | Тянет данные с [Open-Meteo Archive API](https://open-meteo.com/) (координаты Астаны) блоками по 5 лет, склеивает в один DataFrame |
| `models/lstm_model.py` | `LSTM(128) → Dense(1)`, MSE-лосс |
| `pipeline/train_and_predict.py` | Нормализация (`MinMaxScaler`), окно из 30 дней предсказывает следующий день, train/test split без перемешивания (временной ряд) |
| `utils/plot.py` | График «реальная температура vs предсказанная» |

```bash
pip install tensorflow scikit-learn pandas requests matplotlib
python main.py
```
