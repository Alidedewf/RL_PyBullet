import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from models.lstm_model import build_model
from utils.plot import plot_forecast

def prepare_data(df, feature="t_max", seq_len=30):
    values = df[feature].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(len(values_scaled) - seq_len):
        X.append(values_scaled[i:i+seq_len])
        y.append(values_scaled[i+seq_len])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def train_and_predict(df):
    X, y, scaler = prepare_data(df, feature="t_max", seq_len=30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = build_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    last_seq = X[-30:]
    preds_scaled = model.predict(last_seq)
    preds = scaler.inverse_transform(preds_scaled)

    real = df["t_max"].values[-30:]

    plot_forecast(real, preds.flatten())