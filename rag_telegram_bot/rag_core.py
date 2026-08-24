import os, json
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = "index"
EMB_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY")
LLM_MODEL    = os.getenv("LLM_MODEL", "gpt-4o-mini")

if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY не задан в .env")

# --- загрузка индекса в память ---
EMB = np.load(EMB_PATH).astype("float32")             # (N, D)
with open(META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)                                # список словарей

def embed_query(text, model_cache={}):
    """Локальные эмбеддинги через тот же transformer (MiniLM)"""
    from sentence_transformers import SentenceTransformer
    if "model" not in model_cache:
        model_cache["model"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vec = model_cache["model"].encode([text], normalize_embeddings=True)
    return vec.astype("float32")[0]  # (D,)

def top_k(query, k=4):
    """Косинусная близость (у нас эмбеддинги уже нормализованы)"""
    q = embed_query(query)                            # (D,)
    sims = EMB @ q                                    # (N,)
    idx = np.argsort(-sims)[:k]
    results = []
    for i in idx:
        results.append({
            "score": float(sims[i]),
            "meta": META[i]
        })
    return results

def build_prompt(query, passages):
    context_blocks = "\n\n".join(
        f"[{p['meta']['source']} / chunk #{p['meta']['chunk_id']}]\n{p['meta']['text']}"
        for p in passages
    )
    sys = (
        "Ты помощник по документам. Отвечай строго по приведённому контексту. "
        "Если ответа нет в контексте — скажи, что информации нет, и предложи сформулировать вопрос иначе."
    )
    user = (
        f"Вопрос: {query}\n\n"
        f"Контекст (фрагменты из локальных документов):\n{context_blocks}\n\n"
        "Ответ (кратко, по делу, со ссылками на [source/chunk]):"
    )
    return sys, user

def call_llm(system_prompt, user_prompt):
    url = f"{LLM_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def answer(query, k=4):
    passages = top_k(query, k=k)
    sys, user = build_prompt(query, passages)
    return call_llm(sys, user), passages
