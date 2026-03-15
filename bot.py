import os
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from chat_rag import generate_answer_with_context

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
conversation_history = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    conversation_history[chat_id] = []
    await update.message.reply_text("Salam! Mən E-Gov üzrə FAQ chatbotuyam. Elektron hökumət xidmətləri ilə bağlı suallarınızı verə bilərsiniz - sizə kömək etməkdən məmnun olaram.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_query = update.message.text

    if chat_id not in conversation_history:
        conversation_history[chat_id] = []

    history = conversation_history[chat_id]
    answer = await asyncio.to_thread(generate_answer_with_context, user_query, history)
    await update.message.reply_text(answer)
    history.append({"question": user_query, "answer": answer})
    conversation_history[chat_id] = history[-50:]

if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    print("Telegram bot işləyir...")
    app.run_polling()