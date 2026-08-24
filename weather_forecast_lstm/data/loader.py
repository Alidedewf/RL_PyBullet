import requests
import pandas as pd

def load_weather_data(start, end):
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        "latitude=51.1694&longitude=71.4491&"
        f"start_date={start}&end_date={end}&"
        "daily=temperature_2m_max,temperature_2m_min,precipitation_sum&"
        "timezone=Asia/Almaty"
    )
    r = requests.get(url)
    data = r.json()

    if "daily" not in data:
        print("Ошибка: нет ключа 'daily' в ответе", data)
        return None

    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "t_max": data["daily"]["temperature_2m_max"],
        "t_min": data["daily"]["temperature_2m_min"],
        "precip": data["daily"]["precipitation_sum"]
    })
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_full_history(start="2010-01-01", end="2024-12-31"):
    df_all = []
    years = list(range(2010, 2025, 5))
    for y in years:
        y_end = min(y + 4, 2024)
        df = load_weather_data(f"{y}-01-01", f"{y_end}-12-31")
        if df is not None:
            df_all.append(df)
    return pd.concat(df_all).reset_index(drop=True)