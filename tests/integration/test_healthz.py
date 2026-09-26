"""Integration tests for the health endpoint. No external systems involved."""

from fastapi.testclient import TestClient


class TestHealthz:
    """Tests for GET /healthz."""

    def test_returns_ok_status(self, client: TestClient) -> None:
        """Verify the health endpoint responds with a static ok payload."""
        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_post_is_not_allowed(self, client: TestClient) -> None:
        """Verify the health endpoint rejects non-GET methods."""
        response = client.post("/healthz")

        assert response.status_code == 405

    def test_unknown_path_returns_404(self, client: TestClient) -> None:
        """Verify an unregistered path is not served."""
        response = client.get("/does-not-exist")

        assert response.status_code == 404
