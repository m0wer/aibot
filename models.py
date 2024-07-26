import os
from datetime import datetime
from sqlmodel import SQLModel, create_engine, Field as SQLField

from loguru import logger

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/db.sqlite")


class User(SQLModel, table=True):
    id: int = SQLField(primary_key=True)
    telegram_id: int = SQLField(unique=True, index=True)
    system_prompt: str = SQLField(
        default="You are a helpful assistant. "
        "Pay special attention to the most recent messages in the conversation."
    )


class Message(SQLModel, table=True):
    id: int = SQLField(primary_key=True)
    user_id: int = SQLField(foreign_key="user.id")
    content: str
    timestamp: datetime = SQLField(default_factory=datetime.utcnow)
    is_from_user: bool
    is_reset: bool = SQLField(default=False)


# Initialize database
engine = create_engine(DATABASE_URL)
SQLModel.metadata.create_all(engine)
logger.info("Database initialized")
