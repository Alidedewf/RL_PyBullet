import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from rag_core import answer
from dotenv import load_dotenv
import os

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")

WELCOME = (
    "Привет! Я локальный RAG-бот 🤖\n"
    "Пришлите вопрос — я поищу ответ в локальных документах и дополню внешней LLM.\n"
    "Команды: /start /help"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Просто отправьте текст вопроса.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = (update.message.text or "").strip()
    await update.message.chat.send_action("typing")
    try:
        ans, passages = answer(q, k=4)
        refs = "\n".join(f"• {p['meta']['source']} (chunk {p['meta']['chunk_id']})" for p in passages)
        msg = f"{ans}\n\nИсточники:\n{refs}"
        # Телеграм ограничивает длину сообщений ~4096
        await update.message.reply_text(msg[:4000])
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")

async def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
