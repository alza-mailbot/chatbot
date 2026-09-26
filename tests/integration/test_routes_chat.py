"""Integration tests for POST /v1/chat. The Gemini client is replaced by a mock."""

from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from chatbot.config import Settings
from chatbot.core.llm.gemini import GeminiClient, LLMError
from chatbot.main import app

_MAX_BYTES = 64


@pytest.fixture()
def gemini_mock() -> Iterator[AsyncMock]:
    """Install a mocked Gemini client and small-limit settings into the app state.

    Yields:
        AsyncMock: The mock standing in for the shared GeminiClient.
    """
    mock = AsyncMock(spec=GeminiClient)
    mock.generate_reply.return_value = "Generated reply"
    app.state.gemini = mock
    app.state.settings = Settings(
        _env_file=None, gcp_project_id="test-project", max_attachment_bytes=_MAX_BYTES
    )
    try:
        yield mock
    finally:
        del app.state.gemini
        del app.state.settings


class TestChat:
    """Tests for POST /v1/chat."""

    def test_returns_generated_reply(self, client: TestClient, gemini_mock: AsyncMock) -> None:
        """Verify a valid request returns the reply produced by the LLM client."""
        response = client.post("/v1/chat", data={"subject": "Warranty", "body": "My laptop broke."})

        assert response.status_code == 200
        assert response.json() == {"reply": "Generated reply"}
        gemini_mock.generate_reply.assert_awaited_once_with(
            subject="Warranty", body="My laptop broke.", attachments=[]
        )

    def test_subject_is_optional(self, client: TestClient, gemini_mock: AsyncMock) -> None:
        """Verify a request without a subject is accepted."""
        response = client.post("/v1/chat", data={"body": "Hello?"})

        assert response.status_code == 200
        gemini_mock.generate_reply.assert_awaited_once_with(
            subject="", body="Hello?", attachments=[]
        )

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


class TestChatAttachments:
    """Tests for POST /v1/chat with uploaded files."""

    def test_valid_attachment_reaches_the_llm(
        self, client: TestClient, gemini_mock: AsyncMock
    ) -> None:
        """Verify an uploaded file is validated and passed to the LLM client."""
        response = client.post(
            "/v1/chat",
            data={"body": "See attachment."},
            files=[("files", ("doc.pdf", b"%PDF-1.4", "application/pdf"))],
        )

        assert response.status_code == 200
        attachments = gemini_mock.generate_reply.await_args.kwargs["attachments"]
        assert len(attachments) == 1
        assert attachments[0].filename == "doc.pdf"
        assert attachments[0].mime_type == "application/pdf"
        assert attachments[0].data == b"%PDF-1.4"

    def test_multiple_attachments_are_passed_in_order(
        self, client: TestClient, gemini_mock: AsyncMock
    ) -> None:
        """Verify several uploaded files keep their order."""
        response = client.post(
            "/v1/chat",
            data={"body": "Two files."},
            files=[
                ("files", ("a.pdf", b"%PDF-a", "application/pdf")),
                ("files", ("b.png", b"\x89PNG", "image/png")),
            ],
        )

        assert response.status_code == 200
        attachments = gemini_mock.generate_reply.await_args.kwargs["attachments"]
        assert [a.filename for a in attachments] == ["a.pdf", "b.png"]

    def test_unsupported_type_is_rejected(self, client: TestClient, gemini_mock: AsyncMock) -> None:
        """Verify an unsupported file type yields 422 and skips the LLM."""
        response = client.post(
            "/v1/chat",
            data={"body": "Zip inside."},
            files=[("files", ("archive.zip", b"PK", "application/zip"))],
        )

        assert response.status_code == 422
        assert "archive.zip" in response.json()["detail"]
        gemini_mock.generate_reply.assert_not_awaited()

    def test_oversized_attachment_is_rejected(
        self, client: TestClient, gemini_mock: AsyncMock
    ) -> None:
        """Verify a file above the configured limit yields 413 and skips the LLM."""
        response = client.post(
            "/v1/chat",
            data={"body": "Big file."},
            files=[("files", ("big.pdf", b"x" * (_MAX_BYTES + 1), "application/pdf"))],
        )

        assert response.status_code == 413
        gemini_mock.generate_reply.assert_not_awaited()

    def test_empty_attachment_is_rejected(self, client: TestClient, gemini_mock: AsyncMock) -> None:
        """Verify an empty file yields 422 and skips the LLM."""
        response = client.post(
            "/v1/chat",
            data={"body": "Empty file."},
            files=[("files", ("empty.pdf", b"", "application/pdf"))],
        )

        assert response.status_code == 422
        gemini_mock.generate_reply.assert_not_awaited()
