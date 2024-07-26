import os
import asyncio
from datetime import datetime, timedelta
from sqlmodel import Session, select
from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from rq import Queue
from redis import Redis
import ollama
from loguru import logger
from time import time

from worker_tasks import (
    process_message,
    speech_to_text,
    MessageRequest,
    STTRequest,
)
from models import User, Message, engine
from utils import save_message

# Environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/db.sqlite")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
WEBHOOK_PORT = os.getenv("WEBHOOK_PORT")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

logger.debug(
    f"Initialized with REDIS_URL: {REDIS_URL}, DATABASE_URL: {DATABASE_URL}, OLLAMA_URL: {OLLAMA_URL}, OLLAMA_MODEL: {OLLAMA_MODEL}"
)

# Initialize Redis queues
redis_conn = Redis.from_url(REDIS_URL)
default_queue = Queue("default", connection=redis_conn)
high_priority_queue = Queue("high", connection=redis_conn)
gpu_queue = Queue("gpu", connection=redis_conn)
logger.info("Redis queues initialized")

# Ollama client
ollama_client = ollama.Client(host=OLLAMA_URL)
logger.info(f"Ollama client initialized with URL: {OLLAMA_URL}")


# Helper functions
def get_or_create_user(telegram_id: int) -> User:
    with Session(engine) as session:
        user = session.exec(select(User).where(User.telegram_id == telegram_id)).first()
        if not user:
            user = User(telegram_id=telegram_id)
            session.add(user)
            session.commit()
            session.refresh(user)
            logger.info(f"Created new user with telegram_id: {telegram_id}")
        else:
            logger.debug(f"Retrieved existing user with telegram_id: {telegram_id}")
        return user


def get_recent_messages(user_id: int, limit: int = 20) -> list[Message]:
    with Session(engine) as session:
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        messages = session.exec(
            select(Message)
            .where(Message.user_id == user_id)
            .where(Message.timestamp > one_hour_ago)
            .where(Message.is_reset.is_(False))  # type: ignore
            .order_by(Message.timestamp.asc())  # type: ignore
            .limit(limit)
        ).all()
    logger.debug(f"Retrieved {len(messages)} recent messages for user_id: {user_id}")
    return list(messages)


# Command handlers
async def start(update: Update, context):
    user = get_or_create_user(update.effective_user.id)
    welcome_message = f"Welcome! I'm your Ollama-powered assistant using the {OLLAMA_MODEL} model. Send me a message or voice note, and I'll respond."
    await update.message.reply_text(welcome_message)
    logger.info(f"Start command received from user_id: {user.id}")


async def set_prompt(update: Update, context):
    user = get_or_create_user(update.effective_user.id)
    with Session(engine) as session:
        db_user = session.get(User, user.id)
        if db_user:
            current_prompt = db_user.system_prompt
        else:
            current_prompt = "Default system prompt"

    if not context.args:
        await update.message.reply_text(
            f"Current system prompt: {current_prompt}\n\nTo change it, use /prompt followed by the new prompt."
        )
        return

    new_prompt = " ".join(context.args)
    with Session(engine) as session:
        db_user = session.get(User, user.id)
        if db_user:
            db_user.system_prompt = new_prompt
            session.add(db_user)
            session.commit()
    await update.message.reply_text(f"System prompt updated to: {new_prompt}")
    logger.info(f"System prompt updated for user_id: {user.id}")


async def reset_chat(update: Update, context):
    user = get_or_create_user(update.effective_user.id)
    with Session(engine) as session:
        messages_to_reset = session.exec(
            select(Message)
            .where(Message.user_id == user.id)
            .where(Message.is_reset.is_(False))  # type: ignore
        ).all()
        for message in messages_to_reset:
            message.is_reset = True
        session.commit()
    await update.message.reply_text("Chat history has been reset.")
    logger.info(f"Chat history reset for user_id: {user.id}")


# Message handlers
async def handle_text(update: Update, context):
    start_time = time()
    user = get_or_create_user(update.effective_user.id)
    save_message(user.id, update.message.text, True)
    logger.info(
        f"Received text message from user_id: {user.id}. Content: {update.message.text}"
    )

    recent_messages = get_recent_messages(user.id)
    context_messages = [
        f"{'User' if msg.is_from_user else 'Assistant'}: {msg.content}"
        for msg in recent_messages
    ]

    request = MessageRequest(
        user_id=user.id,
        content=update.message.text,
        chat_id=update.effective_chat.id,
        message_id=update.message.message_id,
    )
    logger.debug(f"Enqueueing message processing job for user_id: {user.id}")
    default_queue.enqueue(
        process_message, request, user.system_prompt, context_messages
    )

    # Log the time taken for message processing
    processing_time = time() - start_time
    logger.info(f"Message processing enqueued in {processing_time:.2f} seconds")


async def handle_voice(update: Update, context):
    start_time = time()
    user = get_or_create_user(update.effective_user.id)

    voice = await update.message.voice.get_file()

    voice_file = await voice.download_as_bytearray()
    logger.info(f"Received voice message from user_id: {user.id}")

    is_forwarded: bool = bool(update.message.api_kwargs.get("forward_from"))
    logger.debug(f"Voice message forwarded: {is_forwarded}")

    logger.debug(f"Enqueueing STT job for user_id: {user.id}")
    gpu_queue.enqueue(
        speech_to_text,
        STTRequest(
            audio_file=bytes(voice_file),
            chat_id=update.effective_chat.id,
            message_id=update.message.message_id,
            forwarded=is_forwarded,
        ),
    )

    # Log the time taken for voice message handling
    voice_handling_time = time() - start_time
    logger.info(
        f"Voice message handling completed in {voice_handling_time:.2f} seconds"
    )


async def set_bot_commands(application: Application):
    commands = [
        BotCommand("start", "Start the bot"),
        BotCommand("prompt", "Set a new system prompt"),
        BotCommand("reset", "Reset chat history"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Bot commands have been set")


# Main function
def main():
    logger.info("Starting the Telegram bot")
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("prompt", set_prompt))
    application.add_handler(CommandHandler("reset", reset_chat))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Set bot commands
    asyncio.get_event_loop().run_until_complete(set_bot_commands(application))

    if WEBHOOK_PORT and WEBHOOK_URL:
        logger.info(f"Starting webhook on port {WEBHOOK_PORT}")
        application.run_webhook(
            listen="0.0.0.0",
            port=int(WEBHOOK_PORT),
            url_path=TELEGRAM_TOKEN,
            webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}",
        )
    else:
        logger.info("Telegram bot is now polling for updates")
        application.run_polling()


if __name__ == "__main__":
    main()
