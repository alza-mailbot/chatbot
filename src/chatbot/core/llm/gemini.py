"""Gemini LLM client backed by Vertex AI."""

from collections.abc import Sequence

from google import genai
from google.genai import types

from chatbot.config import Settings
from chatbot.core.attachments import Attachment


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

    async def generate_reply(
        self, subject: str, body: str, attachments: Sequence[Attachment] = ()
    ) -> str:
        """Generate a reply to an email.

        Args:
            subject: Email subject line.
            body: Plain-text email body.
            attachments: Validated attachments included in the prompt.

        Returns:
            str: The generated reply text.

        Raises:
            LLMError: If the model call fails or returns no text.
        """
        contents: list[types.Part | str] = [
            types.Part.from_bytes(data=a.data, mime_type=a.mime_type) for a in attachments
        ]
        contents.append(f"Subject: {subject}\n\n{body}")
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=self._system_prompt),
            )
        except Exception as exc:
            raise LLMError("Gemini request failed") from exc
        if not response.text:
            raise LLMError("Gemini returned an empty response")
        return response.text
