import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TextAnalysisModel:
    """
    🧠 МОДЕЛЬ АНАЛИЗА ТЕКСТА НА TENSORFLOW

    Архитектура: Embedding → LSTM → GlobalMaxPooling → Dense → Output
    Задача: Анализ тональности текста (позитив/негатив)
    """

    def __init__(self, max_words=10000, max_sequence_length=100, embedding_dim=128):
        """
        Инициализация параметров модели

        max_words: размер словаря (самые частые слова)
        max_sequence_length: максимальная длина текста в словах
        embedding_dim: размерность векторного представления слов
        """
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None

    def build_model(self):
        """
        🏗️ ПОСТРОЕНИЕ АРХИТЕКТУРЫ МОДЕЛИ
        """
        print("🏗️ Строим архитектуру модели...")

        self.model = Sequential([
            # 1️⃣ EMBEDDING LAYER - превращает слова в векторы
            Embedding(
                input_dim=self.max_words,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                name='word_embeddings'
            ),

            # 2️⃣ LSTM LAYER - анализирует последовательности и контекст
            LSTM(
                units=64,  # 64 ячейки памяти
                dropout=0.2,  # dropout для входов
                recurrent_dropout=0.2,  # dropout для рекуррентных связей
                name='context_analysis'
            ),

            # 3️⃣ GLOBAL MAX POOLING - извлекает важнейшие признаки
            GlobalMaxPooling1D(name='feature_extraction'),

            # 4️⃣ DENSE HIDDEN LAYER - дополнительная обработка
            Dense(
                units=32,
                activation='relu',
                name='feature_processing'
            ),

            # 5️⃣ DROPOUT - предотвращение переобучения
            Dropout(0.5, name='regularization'),

            # 6️⃣ OUTPUT LAYER - финальная классификация
            Dense(
                units=1,
                activation='sigmoid',  # вероятность от 0 до 1
                name='sentiment_classification'
            )
        ])

        # Компилируем модель с метриками
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        print("✅ Архитектура модели построена!")
        print(f"📊 Параметров в модели: {self.model.count_params():,}")
        return self.model

    def prepare_data(self, texts, labels=None):
        """
        📝 ПОДГОТОВКА ТЕКСТОВЫХ ДАННЫХ

        Процесс:
        1. Токенизация (слова → индексы)
        2. Padding (приведение к одной длине)
        """
        print("📝 Подготавливаем данные...")

        if self.tokenizer is None:
            # Создаем и обучаем токенизатор
            self.tokenizer = Tokenizer(
                num_words=self.max_words,
                oov_token="<UNKNOWN>",  # токен для неизвестных слов
                lower=True,  # приводим к нижнему регистру
                char_level=False  # работаем со словами, не символами
            )

            self.tokenizer.fit_on_texts(texts)
            print(f"📚 Создан словарь из {len(self.tokenizer.word_index)} слов")

            # Показываем примеры самых частых слов
            word_freq = [(word, count) for word, count in
                         self.tokenizer.word_counts.items()]
            word_freq.sort(key=lambda x: x[1], reverse=True)
            print(f"🔤 Топ-5 слов: {word_freq[:5]}")

        # Преобразуем тексты в последовательности индексов
        sequences = self.tokenizer.texts_to_sequences(texts)

        # Приводим все последовательности к одной длине
        sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',  # добавляем нули в конец
            truncating='post'  # обрезаем длинные тексты с конца
        )

        print(f"✅ Подготовлено {len(sequences)} текстов, форма: {sequences.shape}")
        return sequences

    def train(self, texts, labels, validation_split=0.2, epochs=15, batch_size=32):
        """
        🎯 ОБУЧЕНИЕ МОДЕЛИ
        """
        print("🎯 Начинаем обучение модели...")

        # Подготавливаем данные
        X = self.prepare_data(texts, labels)
        y = np.array(labels, dtype=np.float32)

        # Разделяем на train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=42,
            stratify=y
        )

        print(f"📊 Обучающая выборка: {X_train.shape[0]} примеров")
        print(f"📊 Валидационная выборка: {X_val.shape[0]} примеров")
        print(f"📊 Соотношение классов: {np.mean(y_train):.2f} позитивных")

        # Создаем коллбеки для обучения
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.0001,
                verbose=1
            )
        ]

        # Обучаем модель
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Финальная оценка
        final_loss, final_acc, final_precision, final_recall = self.model.evaluate(
            X_val, y_val, verbose=0
        )

        print(f"\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"🎯 Финальная точность: {final_acc:.3f}")
        print(f"🎯 Precision: {final_precision:.3f}")
        print(f"🎯 Recall: {final_recall:.3f}")
        print(f"🎯 F1-Score: {2 * (final_precision * final_recall) / (final_precision + final_recall):.3f}")

        return history

    def predict(self, texts):
        """
        🔮 ПРЕДСКАЗАНИЕ ДЛЯ НОВЫХ ТЕКСТОВ
        """
        if self.model is None:
            raise ValueError("❌ Модель не обучена! Сначала вызовите train()")

        X = self.prepare_data(texts)
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def analyze_text(self, text):
        """
        🧪 ДЕТАЛЬНЫЙ АНАЛИЗ ОДНОГО ТЕКСТА
        """
        prediction = self.predict([text])[0]

        # Интерпретируем результат
        if prediction >= 0.8:
            sentiment = "Очень позитивный"
            emoji = "😍"
            color = "🟢"
        elif prediction >= 0.6:
            sentiment = "Позитивный"
            emoji = "😊"
            color = "🟢"
        elif prediction >= 0.4:
            sentiment = "Нейтральный"
            emoji = "😐"
            color = "🟡"
        elif prediction >= 0.2:
            sentiment = "Негативный"
            emoji = "😔"
            color = "🔴"
        else:
            sentiment = "Очень негативный"
            emoji = "😡"
            color = "🔴"

        confidence = abs(prediction - 0.5) * 2  # уверенность от 0 до 1

        return {
            'text': text,
            'probability': float(prediction),
            'sentiment': sentiment,
            'emoji': emoji,
            'color': color,
            'confidence': float(confidence)
        }

    def batch_analyze(self, texts):
        """
        📋 АНАЛИЗ МНОЖЕСТВА ТЕКСТОВ
        """
        results = []
        predictions = self.predict(texts)

        for text, pred in zip(texts, predictions):
            result = self.analyze_text(text)
            results.append(result)

        return results

    def get_model_info(self):
        """
        📊 ИНФОРМАЦИЯ О МОДЕЛИ
        """
        if self.model is None:
            return "Модель не построена"

        info = {
            'total_params': self.model.count_params(),
            'vocab_size': len(self.tokenizer.word_index) if self.tokenizer else 0,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dim': self.embedding_dim,
            'architecture': [layer.name for layer in self.model.layers]
        }
        return info

    def plot_training_history(self, history):
        """
        📈 ВИЗУАЛИЗАЦИЯ ПРОЦЕССА ОБУЧЕНИЯ
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Точность
        axes[0, 0].plot(history.history['accuracy'], label='Training', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('🎯 Точность модели')
        axes[0, 0].set_xlabel('Эпоха')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Потери
        axes[0, 1].plot(history.history['loss'], label='Training', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('📉 Функция потерь')
        axes[0, 1].set_xlabel('Эпоха')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training', linewidth=2)
        axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('🎯 Precision')
        axes[1, 0].set_xlabel('Эпоха')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('🔍 Recall')
        axes[1, 1].set_xlabel('Эпоха')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_model(self, model_path, tokenizer_path):
        """
        💾 СОХРАНЕНИЕ МОДЕЛИ И ТОКЕНИЗАТОРА
        """
        self.model.save(model_path)

        # Сохраняем токенизатор
        import pickle
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        print(f"💾 Модель сохранена: {model_path}")
        print(f"💾 Токенизатор сохранён: {tokenizer_path}")

    def load_model(self, model_path, tokenizer_path):
        """
        📁 ЗАГРУЗКА МОДЕЛИ И ТОКЕНИЗАТОРА
        """
        self.model = tf.keras.models.load_model(model_path)

        # Загружаем токенизатор
        import pickle
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

        print(f"📁 Модель загружена: {model_path}")
        print(f"📁 Токенизатор загружен: {tokenizer_path}")


def create_sample_data():
    """
    📚 СОЗДАНИЕ ДЕМОНСТРАЦИОННЫХ ДАННЫХ
    """
    positive_texts = [
        "Отличный фильм, рекомендую всем!",
        "Прекрасный день, настроение на высоте",
        "Замечательный ресторан, вкусная еда",
        "Превосходная работа команды, браво!",
        "Великолепное качество товара, доволен покупкой",
        "Чудесная книга, читается на одном дыхании",
        "Потрясающий концерт, незабываемые впечатления",
        "Идеальное обслуживание, быстро и качественно",
        "Восхитительный дизайн, очень стильно",
        "Фантастический результат, превзошел ожидания"
    ]

    negative_texts = [
        "Ужасный фильм, потерянное время",
        "Плохое настроение, все идет не так",
        "Отвратительная еда в ресторане",
        "Кошмарная работа, полный провал",
        "Низкое качество товара, деньги на ветер",
        "Скучная книга, не смог дочитать",
        "Провальный концерт, разочарование полное",
        "Ужасное обслуживание, хамство персонала",
        "Безвкусный дизайн, выглядит дешево",
        "Катастрофический результат, хуже некуда"
    ]

    # Объединяем данные
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)

    return texts, labels


def main():
    """
    🚀 ГЛАВНАЯ ФУНКЦИЯ - ДЕМОНСТРАЦИЯ РАБОТЫ
    """
    print("🚀 ЗАПУСК АНАЛИЗАТОРА ТЕКСТА НА TENSORFLOW")
    print("=" * 60)

    # Создаем модель
    analyzer = TextAnalysisModel(
        max_words=5000,
        max_sequence_length=50,
        embedding_dim=64
    )

    # Строим архитектуру
    model = analyzer.build_model()

    print("\n📋 АРХИТЕКТУРА МОДЕЛИ:")
    print("-" * 40)
    model.summary()

    # Подготавливаем данные
    texts, labels = create_sample_data()

    print(f"\n📚 ДАННЫЕ ДЛЯ ОБУЧЕНИЯ:")
    print(f"📊 Всего примеров: {len(texts)}")
    print(f"📊 Позитивных: {sum(labels)}")
    print(f"📊 Негативных: {len(labels) - sum(labels)}")

    # Обучаем модель
    print(f"\n🎓 НАЧИНАЕМ ОБУЧЕНИЕ...")
    print("-" * 40)

    history = analyzer.train(
        texts=texts,
        labels=labels,
        epochs=20,
        batch_size=4,
        validation_split=0.2
    )

    # Тестируем на новых данных
    test_texts = [
        "Этот продукт просто великолепен, очень доволен!",
        "Полная ерунда, не рекомендую никому",
        "Нормальное качество, ничего особенного",
        "Восхитительный сервис, будем пользоваться еще!",
        "Кошмарный опыт, больше сюда не приду"
    ]

    print(f"\n🧪 ТЕСТИРОВАНИЕ МОДЕЛИ:")
    print("=" * 60)

    for i, text in enumerate(test_texts, 1):
        result = analyzer.analyze_text(text)

        print(f"\n{i}. 📝 Текст: '{result['text']}'")
        print(f"   {result['color']} {result['sentiment']} {result['emoji']}")
        print(f"   📊 Вероятность позитива: {result['probability']:.3f}")
        print(f"   🎯 Уверенность модели: {result['confidence']:.3f}")

    # Пакетный анализ
    print(f"\n📋 ПАКЕТНЫЙ АНАЛИЗ:")
    print("-" * 40)

    batch_results = analyzer.batch_analyze(test_texts)
    for result in batch_results:
        emoji_line = f"{result['emoji']} {result['sentiment']}"
        print(f"{emoji_line:20} | Вероятность: {result['probability']:.3f}")

    # Информация о модели
    print(f"\n📊 ИНФОРМАЦИЯ О МОДЕЛИ:")
    print("-" * 40)
    info = analyzer.get_model_info()
    print(f"🔧 Параметров: {info['total_params']:,}")
    print(f"📚 Размер словаря: {info['vocab_size']:,}")
    print(f"📏 Макс. длина текста: {info['max_sequence_length']}")
    print(f"🔤 Размерность эмбеддингов: {info['embedding_dim']}")

    print(f"\n✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
    print("🎉 Модель готова к использованию в ваших проектах!")

    # Можно раскомментировать для показа графиков
    # analyzer.plot_training_history(history)


if __name__ == "__main__":
    main()