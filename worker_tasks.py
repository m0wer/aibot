import os
from pydub import AudioSegment
from io import BytesIO
from pydantic import BaseModel
import torch
import speech_recognition as sr
from gtts import gTTS
from loguru import logger
import ollama

# Environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/db.sqlite")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# Ollama client
ollama_client = ollama.Client(host=os.getenv("OLLAMA_URL", "http://localhost:11434"))
logger.info(f"Ollama client initialized with URL: {OLLAMA_URL}")


# Pydantic models
class MessageRequest(BaseModel):
    user_id: int
    content: str
    is_audio: bool = False


class TTSRequest(BaseModel):
    text: str


class STTRequest(BaseModel):
    audio_file: bytes


def process_message(
    request: MessageRequest, system_prompt: str, context_messages: list[str]
) -> str:
    logger.debug(f"Processing message for user_id: {request.user_id}")
    messages = [
        {"role": "system", "content": system_prompt},
        *[
            {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
            for i, msg in enumerate(context_messages)
        ],
        {"role": "user", "content": request.content},
    ]
    response = ollama_client.chat(model=OLLAMA_MODEL, messages=messages)
    logger.debug(f"Received response from Ollama for user_id: {request.user_id}")
    return response["message"]["content"]


def text_to_speech(request: TTSRequest) -> bytes:
    logger.debug("Converting text to speech")
    tts = gTTS(text=request.text, lang="en")
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    logger.debug("Text-to-speech conversion completed")
    return audio_file.getvalue()


def convert_ogg_to_wav(ogg_data: bytes) -> bytes:
    """Convert Ogg/Opus audio data to WAV format."""
    audio = AudioSegment.from_file(BytesIO(ogg_data), format="ogg")
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    return wav_io.getvalue()


def speech_to_text(request: STTRequest) -> str:
    logger.debug("Converting speech to text")
    recognizer = sr.Recognizer()
    wav_data = convert_ogg_to_wav(request.audio_file)
    audio_file = BytesIO(wav_data)
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_whisper(
            audio, model="base", device="cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.debug("Speech-to-text conversion completed successfully")
        return text
    except sr.UnknownValueError:
        logger.warning("Speech recognition could not understand the audio")
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        logger.error("Could not request results from speech recognition service")
        return "Sorry, there was an error processing the audio."
