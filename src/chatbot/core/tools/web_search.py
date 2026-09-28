"""Web tools the agent can offer to the model: Brave search and a page reader."""

from html.parser import HTMLParser

import httpx
from pydantic import BaseModel

from chatbot.utils.logger import logger


class ToolError(Exception):
    """A tool failed; the model should answer from what it already has."""


class SearchResult(BaseModel):
    """One simplified web search result.

    Attributes:
        title: Page title.
        url: Page address.
        description: Short result snippet.
    """

    title: str
    url: str
    description: str


class BraveSearchClient:
    """Async client for the Brave Search API web endpoint."""

    def __init__(self, client: httpx.AsyncClient, *, api_key: str, max_results: int = 5) -> None:
        """Wrap a configured HTTP client.

        Args:
            client: AsyncClient with the Brave base_url and timeout set.
            api_key: Brave subscription token, sent with every request.
            max_results: Maximum number of results requested from Brave.
        """
        self._client = client
        self._api_key = api_key
        self._max_results = max_results

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def search(self, query: str) -> list[SearchResult]:
        """Search the web and return simplified results.

        Args:
            query: Search query text.

        Returns:
            list[SearchResult]: Up to max_results results; empty when Brave
                finds nothing.

        Raises:
            ToolError: Transport failure, non-200 status or a malformed body.
        """
        logger.info("[TOOLS] web_search(%r)", query)
        try:
            response = await self._client.get(
                "/res/v1/web/search",
                params={"q": query, "count": self._max_results},
                headers={"X-Subscription-Token": self._api_key},
            )
        except httpx.HTTPError as exc:
            raise ToolError(f"Web search failed: {exc}") from exc
        if response.status_code != 200:
            raise ToolError(f"Web search returned {response.status_code}")
        try:
            items = response.json().get("web", {}).get("results", [])
        except ValueError as exc:
            raise ToolError("Malformed web search response") from exc
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                description=item.get("description", ""),
            )
            for item in items
        ]


class _TextExtractor(HTMLParser):
    """Collects visible text from HTML, skipping script and style content."""

    _SKIPPED_TAGS = frozenset({"script", "style", "noscript", "template"})

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: ARG002
        if tag in self._SKIPPED_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIPPED_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self._chunks).split())


async def fetch_page(client: httpx.AsyncClient, url: str, *, max_chars: int = 8000) -> str:
    """Download a web page and return its visible text.

    Best effort by design: pages behind logins, paywalls or rendered purely
    by JavaScript are reported as failures and the model falls back to the
    search snippets. No authentication is ever attempted.

    Args:
        client: AsyncClient used for the plain GET request.
        url: Absolute address of the page to read.
        max_chars: Length budget for the returned text.

    Returns:
        str: Visible page text, whitespace-collapsed and truncated.

    Raises:
        ToolError: Transport failure, non-200 status, non-HTML content or a
            page with no visible text.
    """
    logger.info("[TOOLS] fetch_page(%r)", url)
    try:
        response = await client.get(url)
    except httpx.HTTPError as exc:
        raise ToolError(f"Page fetch failed: {exc}") from exc
    if response.status_code != 200:
        raise ToolError(f"Page returned {response.status_code}")
    content_type = response.headers.get("content-type", "")
    if not content_type.startswith(("text/html", "text/plain", "application/xhtml")):
        raise ToolError(f"Page has unreadable content type {content_type!r}")
    extractor = _TextExtractor()
    extractor.feed(response.text)
    text = extractor.text()
    if not text:
        raise ToolError("Page contains no readable text")
    return text[:max_chars]
