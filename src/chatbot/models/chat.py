"""Request and response models for the chat API."""

from pydantic import BaseModel


class ChatResponse(BaseModel):
    """Reply generated for an incoming email.

    Attributes:
        reply: Generated plain-text reply.
    """

    reply: str
