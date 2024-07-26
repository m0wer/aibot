from sqlmodel import Session
from loguru import logger

from models import Message, ProcessingTime, engine


def save_message(user_id: int, content: str, is_from_user: bool):
    with Session(engine) as session:
        message = Message(user_id=user_id, content=content, is_from_user=is_from_user)
        session.add(message)
        session.commit()
    logger.debug(f"Saved message for user_id: {user_id}, is_from_user: {is_from_user}")


def save_processing_time(
    user_id: int, operation: str, duration: float, message_id: int | None = None
):
    with Session(engine) as session:
        processing_time = ProcessingTime(
            user_id=user_id,
            operation=operation,
            duration=duration,
            message_id=message_id,
        )
        session.add(processing_time)
        session.commit()
    logger.debug(
        f"Saved processing time for user_id: {user_id}, message_id: {message_id}, operation: {operation}, duration: {duration:.2f} seconds"
    )
