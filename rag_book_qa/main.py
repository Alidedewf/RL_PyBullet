import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ---------- Настройки ----------
BOOK_PATH = "./rules.txt"
INDEX_PATH = "./direct"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN = os.environ["HF_TOKEN"]

TOP_K = 5  # кол-во предложений использовать как контекст

client = InferenceClient(api_key=HF_TOKEN)

# ---------- Чтение текста ----------
def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = " ".join(text.split())
    sentences = [s.strip() for s in text.replace("\n", " ").split('.') if s.strip()]
    return sentences

# ---------- Эмбеддинги ----------
def build_embeddings(sentences, model_name):
    model = SentenceTransformer(model_name)
    vecs = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
    return vecs

def save_index(vecs):
    if not os.path.exists(INDEX_PATH):
        os.makedirs(INDEX_PATH)
    d = vecs.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vecs)
    faiss.write_index(index, f"{INDEX_PATH}/faiss.index")

def load_index():
    return faiss.read_index(f"{INDEX_PATH}/faiss.index")

# ---------- Поиск релевантных предложений ----------
def search(query, model_name, index, sentences, top_k=TOP_K):
    model = SentenceTransformer(model_name)
    q_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec, top_k)
    results = [sentences[i] for i in I[0] if i < len(sentences)]
    return results

# ---------- Формирование ответа через LLM ----------
def ask_llm(question, context_chunks):
    context = " ".join(context_chunks)[:1500]  # ограничение длины контекста
    messages = [
        {"role": "system", "content": "Ты — помощник, отвечающий строго по контексту книги."},
        {"role": "user", "content": (
            f"Контекст книги:\n{context}\n\n"
            f"Вопрос: {question}\n"
            "Ответь одним предложением, точно и корректно, используя только контекст книги."
        )}
    ]
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
        )
        msg = completion.choices[0].message
        if isinstance(msg, dict):
            return msg.get("content", "Пустой ответ.").strip()
        return str(msg).strip()
    except Exception as e:
        return f"Ошибка при запросе модели: {e}"

# ---------- Главная ----------
def main():
    texts_path = f"{INDEX_PATH}/sentences.npy"
    index_path = f"{INDEX_PATH}/faiss.index"

    if not os.path.exists(index_path):
        print("Создаю эмбеддинги...")
        sentences = read_txt(BOOK_PATH)
        vecs = build_embeddings(sentences, EMBED_MODEL)
        np.save(texts_path, np.array(sentences, dtype=object))
        save_index(vecs)
        print("Индекс создан.")
    else:
        print("Загружаю индекс...")

    sentences = np.load(texts_path, allow_pickle=True)
    index = load_index()

    print("RAG по книге готов. Введите вопрос (exit для выхода):")

    while True:
        q = input("\n> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        context_chunks = search(q, EMBED_MODEL, index, sentences)
        if context_chunks:
            answer = ask_llm(q, context_chunks)
            print("\n=== Ответ ===")
            print(answer)
        else:
            print("Ничего не найдено.")

if __name__ == "__main__":
    main()