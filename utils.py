from sqlmodel import Session
from loguru import logger

from models import Message, engine


def save_message(user_id: int, content: str, is_from_user: bool):
    with Session(engine) as session:
        message = Message(user_id=user_id, content=content, is_from_user=is_from_user)
        session.add(message)
        session.commit()
    logger.debug(f"Saved message for user_id: {user_id}, is_from_user: {is_from_user}")
