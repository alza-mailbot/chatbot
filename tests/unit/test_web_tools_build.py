"""Unit tests for building the web tool set from settings."""

from typing import Any

from chatbot.config import Settings
from chatbot.main import build_web_tools


def _settings(**overrides: Any) -> Settings:
    """Build settings with a dummy project and the given overrides."""
    return Settings(_env_file=None, gcp_project_id="test-project", **overrides)


class TestBuildWebTools:
    """Tests for the lifespan helper assembling agent tools."""

    async def test_disabled_flag_builds_nothing(self) -> None:
        """Verify the default settings yield no tools and no clients."""
        tools, clients = build_web_tools(_settings())

        assert tools == {}
        assert clients == []

    async def test_enabled_flag_builds_both_tools(self) -> None:
        """Verify search and page reader are offered once the flag and key are set."""
        tools, clients = build_web_tools(
            _settings(web_search_enabled=True, brave_api_key="test-key")
        )
        try:
            assert set(tools) == {"web_search", "fetch_page"}
            assert all(callable(tool) for tool in tools.values())
        finally:
            for client in clients:
                await client.aclose()
        assert clients
