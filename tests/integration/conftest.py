"""Shared fixtures for integration tests."""

import pytest
from fastapi.testclient import TestClient

from chatbot.main import app


@pytest.fixture()
def client() -> TestClient:
    """Return a test client bound to the application.

    Returns:
        TestClient: Client that exercises the app without a running server.
    """
    return TestClient(app)
