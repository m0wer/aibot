import os
from datetime import datetime
from sqlmodel import SQLModel, create_engine, Field as SQLField, Relationship

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/db.sqlite")


class User(SQLModel, table=True):
    id: int = SQLField(primary_key=True)
    telegram_id: int = SQLField(unique=True, index=True)
    system_prompt: str = SQLField(
        default="You are a helpful assistant running in a telegram bot. "
        "The user will receive your responses in text and as voice messages (TTS). "
        "You will receive the user's text messages directly, and the voice messages transcribed. "
        "Pay more attention to the latest messages. "
        "By default, you will get all messages sent during the last hour, up to a limit."
    )
    messages: list["Message"] = Relationship(back_populates="user")
    processing_times: list["ProcessingTime"] = Relationship(back_populates="user")


class Message(SQLModel, table=True):
    id: int = SQLField(primary_key=True)
    user_id: int = SQLField(foreign_key="user.id")
    content: str
    timestamp: datetime = SQLField(default_factory=datetime.utcnow)
    is_from_user: bool
    is_reset: bool = SQLField(default=False)
    user: User = Relationship(back_populates="messages")


class ProcessingTime(SQLModel, table=True):
    id: int = SQLField(primary_key=True)
    user_id: int = SQLField(foreign_key="user.id")
    message_id: int = SQLField(nullable=True)  # New field for message ID
    timestamp: datetime = SQLField(default_factory=datetime.utcnow)
    operation: str
    duration: float
    user: User = Relationship(back_populates="processing_times")


# Initialize database
engine = create_engine(DATABASE_URL)
