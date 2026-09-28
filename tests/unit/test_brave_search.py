"""Unit tests for the Brave search client. Requests are captured by MockTransport."""

import httpx
import pytest

from chatbot.core.tools.web_search import BraveSearchClient, SearchResult, ToolError

_BRAVE_PAYLOAD = {
    "query": {"original": "alza akcie"},
    "web": {
        "results": [
            {
                "title": "Alza.cz - akcie",
                "url": "https://example.com/alza-akcie",
                "description": "Aktuální kurz akcií.",
                "page_age": "2026-09-28",
            },
            {
                "title": "Burza dnes",
                "url": "https://example.com/burza",
                "description": "Přehled trhu.",
            },
        ]
    },
}


def _client(
    response: httpx.Response | Exception, requests: list[httpx.Request], **kwargs: int
) -> BraveSearchClient:
    """Return a BraveSearchClient whose transport records requests."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if isinstance(response, Exception):
            raise response
        return response

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.search.brave.com"
    )
    return BraveSearchClient(http, api_key="test-key", **kwargs)


class TestSearch:
    """Tests for BraveSearchClient.search request building and parsing."""

    async def test_sends_token_header_and_query(self) -> None:
        """Verify the subscription token and query reach the Brave endpoint."""
        requests: list[httpx.Request] = []
        client = _client(httpx.Response(200, json=_BRAVE_PAYLOAD), requests)

        results = await client.search("alza akcie")

        request = requests[0]
        assert request.url.path == "/res/v1/web/search"
        assert request.headers["X-Subscription-Token"] == "test-key"
        assert request.url.params["q"] == "alza akcie"
        assert results == [
            SearchResult(
                title="Alza.cz - akcie",
                url="https://example.com/alza-akcie",
                description="Aktuální kurz akcií.",
            ),
            SearchResult(
                title="Burza dnes",
                url="https://example.com/burza",
                description="Přehled trhu.",
            ),
        ]

    async def test_requests_at_most_max_results(self) -> None:
        """Verify the configured result cap travels as the count parameter."""
        requests: list[httpx.Request] = []
        client = _client(httpx.Response(200, json=_BRAVE_PAYLOAD), requests, max_results=3)

        await client.search("alza")

        assert requests[0].url.params["count"] == "3"

    async def test_no_results_returns_empty_list(self) -> None:
        """Verify a response without web results is a valid empty outcome."""
        client = _client(httpx.Response(200, json={"query": {"original": "x"}}), [])

        assert await client.search("x") == []


class TestSearchErrors:
    """Tests for error mapping of BraveSearchClient.search."""

    async def test_rate_limit_raises_tool_error(self) -> None:
        """Verify a 429 from Brave maps to ToolError instead of crashing."""
        client = _client(httpx.Response(429, json={"type": "ErrorResponse"}), [])

        with pytest.raises(ToolError):
            await client.search("alza")

    async def test_server_error_raises_tool_error(self) -> None:
        """Verify a 5xx from Brave maps to ToolError."""
        client = _client(httpx.Response(503, text="unavailable"), [])

        with pytest.raises(ToolError):
            await client.search("alza")

    async def test_timeout_raises_tool_error(self) -> None:
        """Verify a network timeout maps to ToolError."""
        client = _client(httpx.ReadTimeout("timed out"), [])

        with pytest.raises(ToolError):
            await client.search("alza")

    async def test_malformed_body_raises_tool_error(self) -> None:
        """Verify a non-JSON 200 response maps to ToolError."""
        client = _client(httpx.Response(200, text="<html>gateway</html>"), [])

        with pytest.raises(ToolError):
            await client.search("alza")
