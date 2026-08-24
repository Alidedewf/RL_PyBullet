import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Загружаем датасет (IMDB отзывы)
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

# Приводим к одинаковой длине
max_length = 256
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)

# Архитектура модели
model = keras.Sequential(name="Text_Classifier")
model.add(layers.Embedding(input_dim=10000, output_dim=32, input_length=max_length, name="Embedding_Layer"))
model.add(layers.GlobalAveragePooling1D(name="Global_Avg_Pool"))
model.add(layers.Dense(16, activation='relu', name="Hidden_Layer"))
model.add(layers.Dense(1, activation='sigmoid', name="Output_Layer"))

# Компиляция
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучение
model.fit(x_train, y_train, epochs=5, batch_size=512, validation_split=0.2)

# Оценка
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Точность на тесте: {accuracy:.2f}")

# Словарь слов → индексы
word_index = keras.datasets.imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

def encode_text(text):
    tokens = text.lower().split()
    encoded = [word_index.get(word, 2) for word in tokens]  # 2 = <UNK>
    padded = keras.preprocessing.sequence.pad_sequences([encoded], maxlen=max_length)
    return padded

# Твой текст
your_text = "I really liked this movie and I'm glad I saw it"
encoded = encode_text(your_text)
x_input = encode_text(your_text)
prediction = model.predict(x_input)[0][0]

print(f"\nТекст: {your_text}")
print(f"Вероятность положительного отзыва: {prediction:.2f}")

