import asyncio
import os
from pydub import AudioSegment
from io import BytesIO
from pydantic import BaseModel
import speech_recognition as sr
from gtts import gTTS
from loguru import logger
import ollama
from time import time
from telegram import Bot
from sqlmodel import Session, select
from utils import save_message, save_processing_time
from models import engine, User

# Environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/db.sqlite")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# Ollama client
ollama_client = ollama.Client(host=OLLAMA_URL)
logger.info(f"Ollama client initialized with URL: {OLLAMA_URL}")

# Telegram bot instance
bot = Bot(token=TELEGRAM_TOKEN)


# Pydantic models
class MessageRequest(BaseModel):
    user_id: int
    content: str
    is_audio: bool = False
    chat_id: int
    message_id: int


class TTSRequest(BaseModel):
    text: str


class TTSResponse(BaseModel):
    audio_data: bytes
    duration: float


class STTRequest(BaseModel):
    audio_file: bytes
    chat_id: int
    message_id: int
    forwarded: bool = False


# Global variable to store the Whisper model
whisper_model = None


async def _process_message(
    request: MessageRequest, system_prompt: str, context_messages: list[str]
) -> None:
    start_time = time()
    logger.debug(f"Processing message for user_id: {request.user_id}")

    if request.is_audio:
        context_messages.append(
            f"User: [The following is a transcription of an audio message from the user] {request.content}"
        )
    else:
        context_messages.append(f"User: {request.content}")

    messages = [
        {"role": "system", "content": system_prompt},
        *[
            {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
            for i, msg in enumerate(context_messages)
        ],
    ]

    ollama_start_time = time()
    response = ollama_client.chat(model=OLLAMA_MODEL, messages=messages, keep_alive=-1)
    ollama_time = time() - ollama_start_time
    save_processing_time(
        request.user_id, "ollama_response", ollama_time, request.message_id
    )
    logger.debug(
        f"Received response from Ollama for user_id: {request.user_id}. Response: {response}"
    )

    # Calculate processing time
    processing_time = time() - start_time

    # Prepare the response text
    response_text = response["message"]["content"]
    response_with_time = (
        f"{response_text}\n\nTotal processing time: {processing_time:.2f} seconds"
    )

    # Save the bot's response to the database
    db_start_time = time()
    with Session(engine) as session:
        user = session.exec(
            select(User).where(User.telegram_id == request.user_id)
        ).first()
        if user:
            save_message(user.id, response_text, False)
    db_time = time() - db_start_time
    save_processing_time(
        request.user_id, "database_operation", db_time, request.message_id
    )

    # Send the text response
    send_start_time = time()
    await bot.send_message(
        chat_id=request.chat_id,
        text=response_with_time,
        reply_to_message_id=request.message_id,
    )
    send_time = time() - send_start_time
    save_processing_time(
        request.user_id, "send_text_response", send_time, request.message_id
    )

    # Generate and send voice message
    if not response_text:
        logger.warning("Empty response text. Skipping text-to-speech conversion")
    else:
        tts_start_time = time()
        tts_response = text_to_speech(TTSRequest(text=response_text))
        tts_time = time() - tts_start_time
        save_processing_time(
            request.user_id, "text_to_speech", tts_time, request.message_id
        )

        voice_send_start_time = time()
        await bot.send_voice(
            chat_id=request.chat_id,
            voice=tts_response.audio_data,
            reply_to_message_id=request.message_id,
            duration=tts_response.duration,
        )
        voice_send_time = time() - voice_send_start_time
        save_processing_time(
            request.user_id, "send_voice_response", voice_send_time, request.message_id
        )

    save_processing_time(
        request.user_id, "total_processing", processing_time, request.message_id
    )
    logger.info(f"Message processed and sent in {processing_time:.2f} seconds")


def process_message(
    request: MessageRequest, system_prompt: str, context_messages: list[str]
) -> None:
    asyncio.run(_process_message(request, system_prompt, context_messages))


def text_to_speech(request: TTSRequest) -> TTSResponse:
    start_time = time()
    logger.debug("Converting text to speech")
    tts = gTTS(text=request.text, lang="en")
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)

    # Get the duration of the audio
    audio = AudioSegment.from_mp3(audio_file)
    duration_seconds = len(audio) / 1000.0

    audio_file.seek(0)
    processing_time = time() - start_time
    logger.debug(
        f"Text-to-speech conversion completed. Duration: {duration_seconds:.2f} seconds, Processing time: {processing_time:.2f} seconds"
    )

    return TTSResponse(audio_data=audio_file.getvalue(), duration=duration_seconds)


def convert_ogg_to_wav(ogg_data: bytes) -> bytes:
    start_time = time()
    logger.debug("Starting conversion from Ogg to WAV")
    audio = AudioSegment.from_file(BytesIO(ogg_data), format="ogg")
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    processing_time = time() - start_time
    logger.debug(
        f"Conversion from Ogg to WAV completed in {processing_time:.2f} seconds"
    )

    return wav_io.getvalue()


async def _speech_to_text(request: STTRequest, **kwargs) -> None:
    global whisper_model
    start_time = time()
    logger.debug("Converting speech to text")
    recognizer = sr.Recognizer()

    # Debug: log the initial size of the audio file
    logger.debug(f"Original Ogg audio file size: {len(request.audio_file)} bytes")

    # Convert Ogg to WAV
    try:
        conversion_start_time = time()
        wav_data = convert_ogg_to_wav(request.audio_file)
        conversion_time = time() - conversion_start_time
        save_processing_time(
            request.chat_id, "audio_conversion", conversion_time, request.message_id
        )
    except Exception as e:
        logger.error(f"Error during Ogg to WAV conversion: {e}")
        await bot.send_message(
            chat_id=request.chat_id,
            text="Sorry, there was an error processing the audio.",
            reply_to_message_id=request.message_id,
        )
        return

    # Debug: log the size of the converted audio file
    logger.debug(f"Converted WAV audio file size: {len(wav_data)} bytes")

    audio_file = BytesIO(wav_data)
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
    except ValueError as e:
        logger.error(f"Error reading audio file: {e}")
        await bot.send_message(
            chat_id=request.chat_id,
            text="Sorry, there was an error reading the audio file.",
            reply_to_message_id=request.message_id,
        )
        return

    try:
        # Use the Whisper model directly without initializing it again
        stt_start_time = time()
        text = recognizer.recognize_whisper(audio_data=audio)
        stt_time = time() - stt_start_time
        save_processing_time(
            request.chat_id, "speech_to_text", stt_time, request.message_id
        )
        processing_time = time() - start_time
        save_processing_time(
            request.chat_id, "total_stt_processing", processing_time, request.message_id
        )
        logger.info(f"Transcribed audio message. Content: {text}")
        logger.debug(
            f"Speech-to-text conversion completed successfully in {processing_time:.2f} seconds"
        )

        # Process the transcribed text
        message_request = MessageRequest(
            user_id=request.chat_id,  # Using chat_id as user_id for simplicity
            content=f"Transcribed audio from user: {text}"
            if request.forwarded
            else f"Transcribed forwarded audio: {text}",
            is_audio=True,
            chat_id=request.chat_id,
            message_id=request.message_id,
        )
        logger.debug(f"Message request: {message_request}")
        await _process_message(
            message_request,
            **kwargs,
        )

    except sr.UnknownValueError:
        logger.warning("Speech recognition could not understand the audio")
        await bot.send_message(
            chat_id=request.chat_id,
            text="Sorry, I couldn't understand the audio.",
            reply_to_message_id=request.message_id,
        )
    except sr.RequestError as e:
        logger.error(f"Could not request results from speech recognition service: {e}")
        await bot.send_message(
            chat_id=request.chat_id,
            text="Sorry, there was an error processing the audio.",
            reply_to_message_id=request.message_id,
        )


def speech_to_text(request: STTRequest, **kwargs) -> None:
    asyncio.run(_speech_to_text(request, **kwargs))
