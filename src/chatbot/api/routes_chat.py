"""Chat endpoints generating email replies."""

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from chatbot.config import Settings
from chatbot.core.attachments import (
    Attachment,
    AttachmentError,
    AttachmentTooLargeError,
    validate_attachment,
)
from chatbot.core.llm.gemini import GeminiClient, LLMError
from chatbot.models.chat import ChatResponse, parse_thread
from chatbot.utils.logger import logger

router = APIRouter()


async def _read_attachments(files: list[UploadFile], *, max_bytes: int) -> list[Attachment]:
    """Read and validate uploaded files.

    Args:
        files: Uploaded multipart files.
        max_bytes: Maximum accepted size of a single file.

    Returns:
        list[Attachment]: Validated attachments in upload order.

    Raises:
        HTTPException: 413 for an oversized file, 422 for an unacceptable one.
    """
    attachments: list[Attachment] = []
    for file in files:
        data = await file.read()
        try:
            attachments.append(
                validate_attachment(
                    file.filename or "attachment",
                    file.content_type or "",
                    data,
                    max_bytes=max_bytes,
                )
            )
        except AttachmentTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except AttachmentError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return attachments


@router.post("/chat")
async def chat(
    request: Request,
    body: Annotated[str, Form()],
    subject: Annotated[str, Form()] = "",
    thread: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
) -> ChatResponse:
    """Generate a reply to an incoming email.

    Args:
        request: Current request, used to access shared app resources.
        body: Plain-text email body.
        subject: Email subject line.
        thread: Prior thread messages as a JSON array of {role, text}.
        files: Optional attachments (PDF, image or audio).

    Returns:
        ChatResponse: The generated reply.

    Raises:
        HTTPException: 413/422 for bad attachments or thread, 502 when the
            LLM fails, 500 on unexpected errors.
    """
    gemini: GeminiClient = request.app.state.gemini
    settings: Settings = request.app.state.settings
    try:
        thread_messages = parse_thread(thread)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid thread field: {exc}") from exc
    attachments = await _read_attachments(files or [], max_bytes=settings.max_attachment_bytes)
    try:
        reply = await gemini.generate_reply(
            subject=subject, body=body, attachments=attachments, thread=thread_messages
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while generating a reply")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(reply=reply)
