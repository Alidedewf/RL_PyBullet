import os, json, glob
import numpy as np
from sentence_transformers import SentenceTransformer

DOC_DIR = "data"
INDEX_DIR = "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800      # символов
CHUNK_OVERLAP = 150   # символов

def read_docs():
    docs = []
    for path in glob.glob(os.path.join(DOC_DIR, "*")):
        if os.path.isfile(path) and path.lower().endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8") as f:
                docs.append((os.path.basename(path), f.read()))
    return docs

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())  # лёгкая нормализация пробелов
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0: start = 0
        if start >= len(text): break
    return chunks

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    docs = read_docs()
    if not docs:
        raise SystemExit(f"Положите .txt/.md файлы в папку {DOC_DIR}/")

    model = SentenceTransformer(MODEL_NAME)
    texts, meta = [], []

    for fname, content in docs:
        chunks = chunk_text(content)
        for i, ch in enumerate(chunks):
            texts.append(ch)
            meta.append({"source": fname, "chunk_id": i, "text": ch})

    print(f"Всего чанков: {len(texts)}")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Индекс сохранён в index/")

if __name__ == "__main__":
    main()
