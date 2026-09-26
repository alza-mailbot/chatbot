"""Live smoke tests against real Vertex AI. Requires ADC and a real GCP project.

Run explicitly with: uv run pytest -m live
"""

import pytest
from dotenv import dotenv_values

from chatbot.config import Settings
from chatbot.core.llm.gemini import GeminiClient


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
