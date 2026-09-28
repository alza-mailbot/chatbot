"""Web tools the agent can offer to the model: Brave search."""

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
