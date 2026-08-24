# lstm_review_classifier

Классификатор тональности текста на LSTM — более развитая версия, чем [`text_classification_imdb`](../text_classification_imdb): обёрнута в класс `TextAnalysisModel` с полным циклом (обучение, сохранение/загрузка, предсказание, визуализация).

**Архитектура:** `Embedding` → `LSTM(64)` → `GlobalMaxPooling1D` → `Dense(32, relu)` → `Dropout(0.5)` → `Dense(1, sigmoid)`, с `EarlyStopping` и `ReduceLROnPlateau`.

Возвращает не просто вероятность, а метку («Очень позитивный» … «Очень негативный») с оценкой уверенности. На небольшом демонстрационном наборе из 20 текстов (на русском) внутри самого файла.

```bash
pip install tensorflow scikit-learn numpy pandas matplotlib
python main.py
```
