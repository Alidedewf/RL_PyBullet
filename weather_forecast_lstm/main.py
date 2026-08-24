from data.loader import load_full_history
from pipeline.train_and_predict import train_and_predict

if __name__ == "__main__":
    df = load_full_history()
    print(df.head())
    train_and_predict(df)