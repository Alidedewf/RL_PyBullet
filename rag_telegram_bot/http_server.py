import os
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# === 1. Загружаем переменные окружения ===
load_dotenv()

API_BASE = os.getenv("LLM_API_BASE")
API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL")

# === 2. Создаём Flask-приложение ===
app = Flask(__name__)

# === 3. Функция общения с LLM ===
def ask_llm(prompt: str):
    # --- OpenAI и совместимые API ---
    if "openai" in API_BASE or "neuraldeep" in API_BASE:
        resp = requests.post(
            f"{API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # --- Gemini API ---
    elif "googleapis" in API_BASE:
        url = f"{API_BASE}/models/{MODEL}:generateContent?key={API_KEY}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]
        }

        resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)

        if resp.status_code != 200:
            print("❌ Ошибка Gemini:", resp.text)
            return f"Ошибка Gemini API ({resp.status_code}): {resp.text}"

        data = resp.json()

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return f"Ошибка в ответе Gemini: {data}"

    # --- Неизвестный API ---
    else:
        raise ValueError(f"Неизвестный API_BASE: {API_BASE}")

# === 4. HTTP эндпоинт ===
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        answer = ask_llm(query)
        return jsonify({"query": query, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === 5. Запуск сервера ===
if __name__ == "__main__":
    print("✅ Сервер запущен: http://127.0.0.1:8008/ask")
    app.run(host="0.0.0.0", port=8008)
