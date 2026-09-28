"""Unit tests for the page reader tool. Requests are captured by MockTransport."""

import httpx
import pytest

from chatbot.core.tools.web_search import ToolError, fetch_page

_HTML = b"""
<html><head><title>Alza akcie</title>
<style>body { color: red; }</style>
<script>console.log("tracker");</script>
</head>
<body>
  <h1>Kurz akci\xc3\xad</h1>
  <p>Dne\xc5\xa1n\xc3\xad    cena je
  1234 K\xc4\x8d.</p>
</body></html>
"""


def _http(response: httpx.Response | Exception, requests: list[httpx.Request]) -> httpx.AsyncClient:
    """Return an AsyncClient whose transport records requests."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if isinstance(response, Exception):
            raise response
        return response

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestFetchPage:
    """Tests for fetch_page content extraction."""

    async def test_returns_visible_text_without_markup(self) -> None:
        """Verify HTML is reduced to readable text, dropping scripts and styles."""
        requests: list[httpx.Request] = []
        http = _http(
            httpx.Response(200, content=_HTML, headers={"content-type": "text/html"}), requests
        )

        text = await fetch_page(http, "https://example.com/alza-akcie")

        assert str(requests[0].url) == "https://example.com/alza-akcie"
        assert "Kurz akcií" in text
        assert "1234 Kč" in text
        assert "<p>" not in text
        assert "console.log" not in text
        assert "color: red" not in text

    async def test_collapses_whitespace(self) -> None:
        """Verify runs of whitespace do not waste the length budget."""
        http = _http(httpx.Response(200, content=_HTML, headers={"content-type": "text/html"}), [])

        text = await fetch_page(http, "https://example.com/x")

        assert "Dnešní cena je 1234 Kč." in text

    async def test_truncates_to_max_chars(self) -> None:
        """Verify overly long pages are cut to the configured budget."""
        body = b"<html><body><p>" + b"slovo " * 5000 + b"</p></body></html>"
        http = _http(httpx.Response(200, content=body, headers={"content-type": "text/html"}), [])

        text = await fetch_page(http, "https://example.com/long", max_chars=100)

        assert len(text) <= 100


class TestFetchPageErrors:
    """Tests for the best-effort error mapping of fetch_page."""

    @pytest.mark.parametrize("status", [401, 403, 404, 500])
    async def test_error_status_raises_tool_error(self, status: int) -> None:
        """Verify auth walls and failures map to ToolError instead of crashing."""
        http = _http(httpx.Response(status, text="denied"), [])

        with pytest.raises(ToolError):
            await fetch_page(http, "https://example.com/private")

    async def test_timeout_raises_tool_error(self) -> None:
        """Verify a network timeout maps to ToolError."""
        http = _http(httpx.ReadTimeout("timed out"), [])

        with pytest.raises(ToolError):
            await fetch_page(http, "https://example.com/slow")

    async def test_non_html_content_raises_tool_error(self) -> None:
        """Verify binary content is refused rather than fed to the model."""
        http = _http(
            httpx.Response(200, content=b"\x89PNG...", headers={"content-type": "image/png"}), []
        )

        with pytest.raises(ToolError):
            await fetch_page(http, "https://example.com/logo.png")

    async def test_blank_page_raises_tool_error(self) -> None:
        """Verify a page with no visible text is treated as a failed fetch."""
        http = _http(
            httpx.Response(
                200,
                content=b"<html><script>app()</script></html>",
                headers={"content-type": "text/html"},
            ),
            [],
        )

        with pytest.raises(ToolError):
            await fetch_page(http, "https://example.com/spa")
