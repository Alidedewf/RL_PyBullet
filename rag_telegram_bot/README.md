# rag_telegram_bot

Telegram-бот, отвечающий на вопросы по локальной базе документов (RAG — Retrieval-Augmented Generation), с ответами через внешнюю LLM (Gemini по умолчанию, любой OpenAI-совместимый API).

| Файл | Роль |
|---|---|
| `build_index.py` | Читает `.txt`/`.md` из `data/`, режет на чанки (800 симв., overlap 150), считает эмбеддинги (`all-MiniLM-L6-v2`), сохраняет в `index/embeddings.npy` + `index/meta.json` |
| `rag_core.py` | Поиск ближайших чанков по косинусной близости + сборка промпта + запрос к LLM |
| `telegram_bot.py` | Точка входа — Telegram-бот на `python-telegram-bot`, использует `rag_core.answer()` |
| `http_server.py` | Альтернативный интерфейс — тот же LLM-чат, но как Flask HTTP API |
| `index/` | Уже посчитанный индекс (эмбеддинги + метаданные чанков) |

```bash
pip install -r requirements.txt sentence-transformers
cp .env.example .env   # заполнить TELEGRAM_BOT_TOKEN и LLM_API_KEY
python build_index.py  # пересчитать индекс из своих документов (опционально — index/ уже есть)
python telegram_bot.py
```

> `.env` с реальными ключами никогда не коммитится (см. `.gitignore`) — при клонировании нужно завести свой.
