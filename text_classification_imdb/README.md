# text_classification_imdb

Классификация тональности отзывов IMDB (позитивный/негативный) на Keras.

- `Embedding(10000, 32)` → `GlobalAveragePooling1D` → `Dense(16, relu)` → `Dense(1, sigmoid)`
- Датасет — встроенный `keras.datasets.imdb`, 10 000 самых частых слов, отзывы обрезаны/дополнены до 256 токенов
- После обучения (5 эпох) модель проверяется на собственном тексте — кодирует его тем же словарём и выводит вероятность позитивного отзыва

```bash
pip install tensorflow numpy
python text_classification.py
```
