"""Chat endpoints generating email replies."""

from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request

from chatbot.core.llm.gemini import GeminiClient, LLMError
from chatbot.models.chat import ChatResponse
from chatbot.utils.logger import logger

router = APIRouter()


@router.post("/chat")
async def chat(
    request: Request,
    body: Annotated[str, Form()],
    subject: Annotated[str, Form()] = "",
) -> ChatResponse:
    """Generate a reply to an incoming email.

    Args:
        request: Current request, used to access the shared Gemini client.
        body: Plain-text email body.
        subject: Email subject line.

    Returns:
        ChatResponse: The generated reply.

    Raises:
        HTTPException: 502 when the LLM fails, 500 on unexpected errors.
    """
    gemini: GeminiClient = request.app.state.gemini
    try:
        reply = await gemini.generate_reply(subject=subject, body=body)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while generating a reply")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(reply=reply)
