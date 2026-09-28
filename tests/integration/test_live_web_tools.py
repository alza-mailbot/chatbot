"""Live smoke tests against the real Brave API and the open web.

Run explicitly with: uv run pytest -m live
"""

import httpx
import pytest
from dotenv import dotenv_values

from chatbot.core.tools.web_search import BraveSearchClient, fetch_page

pytestmark = pytest.mark.live


def _brave_key() -> str:
    """Return the real Brave key from the local .env or skip the test."""
    key = dotenv_values(".env").get("BRAVE_API_KEY")
    if not key:
        pytest.skip("No BRAVE_API_KEY configured in .env")
    return key


class TestLiveBraveSearch:
    """Smoke tests for the real Brave endpoint."""

    async def test_search_returns_populated_results(self) -> None:
        """Verify a real query yields results with title, url and description."""
        http = httpx.AsyncClient(base_url="https://api.search.brave.com", timeout=10)
        client = BraveSearchClient(http, api_key=_brave_key(), max_results=3)
        try:
            results = await client.search("Alza.cz")
        finally:
            await client.aclose()

        assert results
        assert len(results) <= 3
        for result in results:
            assert result.title
            assert result.url.startswith("http")


class TestLiveFetchPage:
    """Smoke tests for the page reader against a stable public page."""

    async def test_reads_example_domain(self) -> None:
        """Verify a real page is reduced to its visible text."""
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as http:
            text = await fetch_page(http, "https://example.com/")

        assert "Example Domain" in text
        assert "<" not in text
