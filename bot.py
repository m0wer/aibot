import os
import asyncio
from typing import List
from datetime import datetime
from sqlmodel import SQLModel, Field as SQLField, create_engine, Session, select
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from rq import Queue
from redis import Redis
import ollama
from loguru import logger
from io import BytesIO

from worker_tasks import (
    process_message,
    text_to_speech,
    speech_to_text,
    MessageRequest,
    TTSRequest,
    STTRequest,
    TTSResponse,
)

# Environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/db.sqlite")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

logger.debug(
    f"Initialized with REDIS_URL: {REDIS_URL}, DATABASE_URL: {DATABASE_URL}, OLLAMA_URL: {OLLAMA_URL}, OLLAMA_MODEL: {OLLAMA_MODEL}"
)


# Database models
class User(SQLModel, table=True):
    id: int = SQLField(primary_key=True)
    telegram_id: int = SQLField(unique=True, index=True)
    system_prompt: str = SQLField(default="You are a helpful assistant.")


class Message(SQLModel, table=True):
    id: int = SQLField(primary_key=True)
    user_id: int = SQLField(foreign_key="user.id")
    content: str
    timestamp: datetime = SQLField(default_factory=datetime.utcnow)
    is_from_user: bool


# Initialize database
engine = create_engine(DATABASE_URL)
SQLModel.metadata.create_all(engine)
logger.info("Database initialized")

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


def save_message(user_id: int, content: str, is_from_user: bool):
    with Session(engine) as session:
        message = Message(user_id=user_id, content=content, is_from_user=is_from_user)
        session.add(message)
        session.commit()
    logger.debug(f"Saved message for user_id: {user_id}, is_from_user: {is_from_user}")


def get_recent_messages(user_id: int, limit: int = 5) -> List[Message]:
    with Session(engine) as session:
        messages = session.exec(
            select(Message)
            .where(Message.user_id == user_id)
            .order_by(Message.timestamp.desc())  # type: ignore
            .limit(limit)
        ).all()
    logger.debug(f"Retrieved {len(messages)} recent messages for user_id: {user_id}")
    return list(messages)


async def wait_for_job_result(job, timeout=60):
    start_time = datetime.now()
    while (datetime.now() - start_time).seconds < timeout:
        if job.result is not None:
            return job.result
        await asyncio.sleep(0.1)
    raise TimeoutError("Job result timeout")


# Command handlers
async def start(update: Update, context):
    user = get_or_create_user(update.effective_user.id)
    welcome_message = f"Welcome! I'm your Ollama-powered assistant using the {OLLAMA_MODEL} model. Send me a message or voice note, and I'll respond."
    await update.message.reply_text(welcome_message)
    logger.info(f"Start command received from user_id: {user.id}")


async def set_system_prompt(update: Update, context):
    user = get_or_create_user(update.effective_user.id)
    new_prompt = " ".join(context.args)
    with Session(engine) as session:
        user.system_prompt = new_prompt
        session.add(user)
        session.commit()
    await update.message.reply_text(f"System prompt updated to: {new_prompt}")
    logger.info(f"System prompt updated for user_id: {user.id}")


# Message handlers
async def handle_text(update: Update, context):
    user = get_or_create_user(update.effective_user.id)
    save_message(user.id, update.message.text, True)
    logger.info(f"Received text message from user_id: {user.id}")

    recent_messages = get_recent_messages(user.id)
    context_messages = [
        f"{'User' if msg.is_from_user else 'Assistant'}: {msg.content}"
        for msg in reversed(recent_messages)
    ]

    request = MessageRequest(user_id=user.id, content=update.message.text)
    logger.debug(f"Enqueueing message processing job for user_id: {user.id}")
    job = default_queue.enqueue(
        process_message, request, user.system_prompt, context_messages
    )

    try:
        response = await wait_for_job_result(job)
    except TimeoutError:
        response = "Sorry, I couldn't process your message in time."
        logger.error(
            f"Timeout waiting for message processing job result for user_id: {user.id}"
        )

    save_message(user.id, response, False)
    logger.debug(f"Received response for user_id: {user.id}")

    await update.message.reply_text(response)

    logger.debug(f"Enqueueing TTS job for user_id: {user.id}")
    tts_job = high_priority_queue.enqueue(text_to_speech, TTSRequest(text=response))

    try:
        tts_result = await wait_for_job_result(tts_job)
        if isinstance(tts_result, TTSResponse):
            tts_response = tts_result
        else:
            logger.error(f"Unexpected TTS job result type for user_id: {user.id}")
            tts_response = None
    except TimeoutError:
        logger.error(f"Timeout waiting for TTS job result for user_id: {user.id}")
        tts_response = None

    if tts_response:
        audio_file = InputFile(
            BytesIO(tts_response.audio_data), filename="voice_message.ogg"
        )
        await update.message.reply_voice(
            audio_file, duration=int(tts_response.duration)
        )
        logger.info(
            f"Sent voice response to user_id: {user.id} with duration: {tts_response.duration:.2f} seconds"
        )
    else:
        logger.warning(f"Failed to generate voice message for user_id: {user.id}")


async def handle_voice(update: Update, context):
    user = get_or_create_user(update.effective_user.id)
    voice = await update.message.voice.get_file()
    voice_file = await voice.download_as_bytearray()
    logger.info(f"Received voice message from user_id: {user.id}")

    logger.debug(f"Enqueueing STT job for user_id: {user.id}")
    stt_job = gpu_queue.enqueue(
        speech_to_text, STTRequest(audio_file=bytes(voice_file))
    )

    try:
        transcribed_text = await wait_for_job_result(stt_job)
    except TimeoutError:
        transcribed_text = "Sorry, I couldn't transcribe your voice message in time."
        logger.error(f"Timeout waiting for STT job result for user_id: {user.id}")

    logger.debug(f"Transcribed text for user_id: {user.id}: {transcribed_text}")

    save_message(user.id, transcribed_text, True)

    recent_messages = get_recent_messages(user.id)
    context_messages = [
        f"{'User' if msg.is_from_user else 'Assistant'}: {msg.content}"
        for msg in reversed(recent_messages)
    ]

    request = MessageRequest(user_id=user.id, content=transcribed_text)
    logger.debug(f"Enqueueing message processing job for user_id: {user.id}")
    job = default_queue.enqueue(
        process_message, request, user.system_prompt, context_messages
    )

    try:
        response = await wait_for_job_result(job)
    except TimeoutError:
        response = "Sorry, I couldn't process your message in time."
        logger.error(
            f"Timeout waiting for message processing job result for user_id: {user.id}"
        )

    save_message(user.id, response, False)
    logger.debug(f"Received response for user_id: {user.id}")

    await update.message.reply_text(response)

    logger.debug(f"Enqueueing TTS job for user_id: {user.id}")
    tts_job = high_priority_queue.enqueue(text_to_speech, TTSRequest(text=response))

    try:
        tts_result = await wait_for_job_result(tts_job)
        if isinstance(tts_result, TTSResponse):
            tts_response = tts_result
        else:
            logger.error(f"Unexpected TTS job result type for user_id: {user.id}")
            tts_response = None
    except TimeoutError:
        logger.error(f"Timeout waiting for TTS job result for user_id: {user.id}")
        tts_response = None

    if tts_response:
        audio_file = InputFile(
            BytesIO(tts_response.audio_data), filename="voice_message.ogg"
        )
        await update.message.reply_voice(
            audio_file, duration=int(tts_response.duration)
        )
        logger.info(
            f"Sent voice response to user_id: {user.id} with duration: {tts_response.duration:.2f} seconds"
        )
    else:
        await update.message.reply_text(
            "Sorry, I couldn't generate the voice message. Here's the text response instead."
        )


# Main function
def main():
    logger.info("Starting the Telegram bot")
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("set_system_prompt", set_system_prompt))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))

    logger.info("Telegram bot is now polling for updates")
    application.run_polling()


if __name__ == "__main__":
    main()
