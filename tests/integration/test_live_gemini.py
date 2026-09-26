"""Live smoke tests against real Vertex AI. Requires ADC and a real GCP project.

Run explicitly with: uv run pytest -m live
"""

from pathlib import Path

import pytest
from dotenv import dotenv_values

from chatbot.config import Settings
from chatbot.core.attachments import Attachment
from chatbot.core.llm.gemini import GeminiClient

_FIXTURES = Path(__file__).parent.parent / "fixtures"


def _real_settings() -> Settings:
    """Build settings from the local .env, bypassing the test-wide dummy project id.

    Returns:
        Settings: Settings pointing at the real GCP project.
    """
    project_id = dotenv_values(".env").get("GCP_PROJECT_ID")
    if not project_id:
        pytest.skip("No real GCP_PROJECT_ID configured in .env")
    return Settings(gcp_project_id=project_id)


@pytest.mark.live
class TestLiveGemini:
    """Smoke tests exercising the real Gemini model."""

    async def test_generates_nonempty_reply(self) -> None:
        """Verify a real Gemini call produces a non-empty text reply."""
        client = GeminiClient(_real_settings())

        reply = await client.generate_reply(
            subject="Greeting", body="Reply with a short greeting in English."
        )

        assert reply.strip()

    async def test_reads_pdf_attachment(self) -> None:
        """Verify Gemini extracts information from an attached PDF."""
        client = GeminiClient(_real_settings())
        pdf = Attachment(
            filename="sample.pdf",
            mime_type="application/pdf",
            data=(_FIXTURES / "sample.pdf").read_bytes(),
        )

        reply = await client.generate_reply(
            subject="Document check",
            body="What is the secret word in the attached document? Answer with just the word.",
            attachments=[pdf],
        )

        assert "PINEAPPLE" in reply.upper()
