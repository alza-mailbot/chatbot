"""Gemini LLM client backed by Vertex AI."""

from collections.abc import Sequence
from datetime import date

from google import genai
from google.genai import types

from chatbot.config import Settings
from chatbot.core.attachments import Attachment
from chatbot.models.chat import ThreadMessage


class LLMError(Exception):
    """Raised when the LLM fails to produce a usable reply."""


class GeminiClient:
    """Async wrapper around the google-genai SDK for email reply generation."""

    def __init__(self, settings: Settings) -> None:
        """Create the underlying Vertex AI client.

        Args:
            settings: Application settings providing project, location and model.
        """
        self._client = genai.Client(
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )
        self._model = settings.gemini_model
        self._system_prompt = settings.system_prompt

    async def generate(
        self,
        contents: list[types.Content],
        *,
        tools: list[types.Tool] | None = None,
    ) -> types.GenerateContentResponse:
        """Run one model call over an explicit conversation.

        Args:
            contents: Full conversation so far, oldest turn first.
            tools: Function declarations offered to the model, if any.

        Returns:
            types.GenerateContentResponse: The whole response; it may carry
                text or a function call, which is for the caller to decide.

        Raises:
            LLMError: If the model call fails.
        """
        try:
            return await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    # the model has no clock; without today's date it cannot
                    # tell current events from its training-data past
                    system_instruction=(
                        f"{self._system_prompt}\nToday's date: {date.today().isoformat()}."
                    ),
                    tools=tools,
                ),
            )
        except Exception as exc:
            raise LLMError("Gemini request failed") from exc

    async def generate_reply(
        self,
        subject: str,
        body: str,
        attachments: Sequence[Attachment] = (),
        thread: Sequence[ThreadMessage] = (),
    ) -> str:
        """Generate a reply to an email in a single model call.

        Args:
            subject: Email subject line.
            body: Plain-text email body.
            attachments: Validated attachments included in the prompt.
            thread: Prior thread messages in chronological order.

        Returns:
            str: The generated reply text.

        Raises:
            LLMError: If the model call fails or returns no text.
        """
        contents = build_contents(
            subject=subject, body=body, attachments=attachments, thread=thread
        )
        response = await self.generate(contents)
        if not response.text:
            raise LLMError("Gemini returned an empty response")
        return response.text


def build_contents(
    *,
    subject: str,
    body: str,
    attachments: Sequence[Attachment] = (),
    thread: Sequence[ThreadMessage] = (),
) -> list[types.Content]:
    """Build the opening conversation for an email reply.

    Args:
        subject: Email subject line.
        body: Plain-text email body.
        attachments: Validated attachments included in the prompt.
        thread: Prior thread messages in chronological order.

    Returns:
        list[types.Content]: Thread history followed by the current user
            turn, attachments first and the email text last.
    """
    history = [
        types.Content(
            role="user" if message.role == "user" else "model",
            parts=[types.Part.from_text(text=message.text)],
        )
        for message in thread
    ]
    current_parts = [types.Part.from_bytes(data=a.data, mime_type=a.mime_type) for a in attachments]
    current_parts.append(types.Part.from_text(text=f"Subject: {subject}\n\n{body}"))
    return [*history, types.Content(role="user", parts=current_parts)]
