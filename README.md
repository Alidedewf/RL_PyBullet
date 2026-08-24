# ml-coursework

Учебные ML-проекты вне линии Reinforcement Learning (см. ветки [`main`](../../tree/main), [`dev`](../../tree/dev), [`rl_final`](../../tree/rl_final) этого репозитория) — классификация текста и изображений, LSTM для временных рядов, два RAG-бота.

| Проект | Задача | Стек |
|---|---|---|
| [text_classification_imdb](text_classification_imdb) | Классификация тональности отзывов (positive/negative) | TensorFlow/Keras, Embedding + GlobalAveragePooling |
| [emnist_digit_classifier](emnist_digit_classifier) | Распознавание рукописных цифр/букв (EMNIST) | PyTorch, CNN |
| [lstm_review_classifier](lstm_review_classifier) | Классификация текста через LSTM | TensorFlow/Keras |
| [weather_forecast_lstm](weather_forecast_lstm) | Прогноз макс. температуры в Астане по истории погоды (Open-Meteo API) | TensorFlow/Keras LSTM, модульная структура (data/pipeline/models/utils) |
| [rag_telegram_bot](rag_telegram_bot) | Telegram-бот, отвечающий по локальной базе знаний + внешняя LLM (Gemini) | RAG, python-telegram-bot, Flask |
| [rag_book_qa](rag_book_qa) | RAG по тексту книги — вопрос-ответ через FAISS + Hugging Face Inference | sentence-transformers, faiss, Mistral-7B |

Каждый проект — в отдельной папке со своим README.
