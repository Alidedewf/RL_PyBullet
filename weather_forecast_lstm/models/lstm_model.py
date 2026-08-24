from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_model(input_shape):
    model = Sequential([
        LSTM(128, activation="tanh", input_shape=input_shape),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model