"""Integration tests for POST /v1/chat. The Gemini client is replaced by a mock."""

from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from chatbot.core.llm.gemini import GeminiClient, LLMError
from chatbot.main import app


@pytest.fixture()
def gemini_mock() -> Iterator[AsyncMock]:
    """Install a mocked Gemini client into the app state.

    Yields:
        AsyncMock: The mock standing in for the shared GeminiClient.
    """
    mock = AsyncMock(spec=GeminiClient)
    mock.generate_reply.return_value = "Generated reply"
    app.state.gemini = mock
    try:
        yield mock
    finally:
        del app.state.gemini


class TestChat:
    """Tests for POST /v1/chat."""

    def test_returns_generated_reply(self, client: TestClient, gemini_mock: AsyncMock) -> None:
        """Verify a valid request returns the reply produced by the LLM client."""
        response = client.post("/v1/chat", data={"subject": "Warranty", "body": "My laptop broke."})

        assert response.status_code == 200
        assert response.json() == {"reply": "Generated reply"}
        gemini_mock.generate_reply.assert_awaited_once_with(
            subject="Warranty", body="My laptop broke."
        )

    def test_subject_is_optional(self, client: TestClient, gemini_mock: AsyncMock) -> None:
        """Verify a request without a subject is accepted."""
        response = client.post("/v1/chat", data={"body": "Hello?"})

        assert response.status_code == 200
        gemini_mock.generate_reply.assert_awaited_once_with(subject="", body="Hello?")

    def test_missing_body_is_rejected(self, client: TestClient, gemini_mock: AsyncMock) -> None:
        """Verify a request without a body fails validation and skips the LLM."""
        response = client.post("/v1/chat", data={"subject": "Warranty"})

        assert response.status_code == 422
        gemini_mock.generate_reply.assert_not_awaited()

    def test_llm_error_maps_to_502(self, client: TestClient, gemini_mock: AsyncMock) -> None:
        """Verify an LLM failure is reported as a bad gateway."""
        gemini_mock.generate_reply.side_effect = LLMError("Gemini request failed")

        response = client.post("/v1/chat", data={"body": "Hello?"})

        assert response.status_code == 502
        assert "Gemini request failed" in response.json()["detail"]

    def test_unexpected_error_maps_to_500(
        self, client: TestClient, gemini_mock: AsyncMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify an unexpected failure is logged and reported as a server error."""
        gemini_mock.generate_reply.side_effect = RuntimeError("boom")

        with caplog.at_level("ERROR", logger="chatbot"):
            response = client.post("/v1/chat", data={"body": "Hello?"})

        assert response.status_code == 500
        assert "Unexpected error" in caplog.text
