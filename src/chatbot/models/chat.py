"""Request and response models for the chat API."""

from typing import Literal

from pydantic import BaseModel, TypeAdapter


class ChatResponse(BaseModel):
    """Reply generated for an incoming email.

    Attributes:
        reply: Generated plain-text reply.
    """

    reply: str


class ThreadMessage(BaseModel):
    """One prior message in an email thread.

    Attributes:
        role: Author of the message: the customer (user) or the bot (assistant).
        text: Plain-text content of the message.
    """

    role: Literal["user", "assistant"]
    text: str


_THREAD_ADAPTER = TypeAdapter(list[ThreadMessage])


def parse_thread(raw: str | None) -> list[ThreadMessage]:
    """Parse the thread form field into message objects.

    Args:
        raw: JSON array of messages, or None/empty when no history is sent.

    Returns:
        list[ThreadMessage]: Parsed messages in chronological order.

    Raises:
        ValueError: If the value is not valid JSON or has a wrong shape
            (pydantic's ValidationError is a ValueError subclass).
    """
    if not raw:
        return []
    return _THREAD_ADAPTER.validate_json(raw)
