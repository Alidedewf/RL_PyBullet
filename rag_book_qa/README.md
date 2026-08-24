# rag_book_qa

RAG по тексту книги (`rules.txt`) — консольный вопрос-ответ: локальный поиск по FAISS + ответ через Hugging Face Inference API (Mistral-7B-Instruct).

- `read_txt` бьёт текст на предложения
- `build_embeddings` / `save_index` — эмбеддинги через `sentence-transformers/all-MiniLM-L6-v2`, индекс — `faiss.IndexFlatL2` (уже посчитан, лежит в `direct/`)
- `search` находит top-5 ближайших предложений к вопросу
- `ask_llm` собирает промпт «отвечай строго по контексту» и шлёт в Mistral-7B через `InferenceClient`

```bash
pip install faiss-cpu sentence-transformers huggingface_hub numpy
export HF_TOKEN=hf_...   # токен Hugging Face
python main.py
```

Индекс (`direct/faiss.index`, `direct/sentences.npy`) уже посчитан для `rules.txt` — при первом запуске просто загружается, пересчитывается только если файлов нет.
