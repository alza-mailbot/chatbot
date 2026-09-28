"""Application entry point building the FastAPI app."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import Protocol

import httpx
from fastapi import FastAPI

from chatbot.api.routes_chat import router as chat_router
from chatbot.api.routes_health import router as health_router
from chatbot.config import Settings, get_settings
from chatbot.core.agent import Tool
from chatbot.core.llm.gemini import GeminiClient
from chatbot.core.tools.web_search import BraveSearchClient, fetch_page
from chatbot.utils.logger import logger


class _Closable(Protocol):
    """Anything owning network resources that must be released on shutdown."""

    async def aclose(self) -> None:
        """Release the underlying resources."""


def build_web_tools(settings: Settings) -> tuple[dict[str, Tool], list[_Closable]]:
    """Assemble the agent tool set the configuration asks for.

    Args:
        settings: Application settings deciding whether web search is on.

    Returns:
        tuple: Executable tools by name (empty when web search is off) and
            the clients to close on shutdown.
    """
    if not (settings.web_search_enabled and settings.brave_api_key):
        return {}, []
    brave = BraveSearchClient(
        httpx.AsyncClient(base_url="https://api.search.brave.com", timeout=10),
        api_key=settings.brave_api_key,
        max_results=settings.brave_max_results,
    )
    page_client = httpx.AsyncClient(timeout=10, follow_redirects=True)
    tools: dict[str, Tool] = {
        "web_search": brave.search,
        "fetch_page": partial(fetch_page, page_client),
    }
    return tools, [brave, page_client]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared resources on startup and log the application's lifetime.

    Args:
        app: The application whose state receives the shared resources.

    Yields:
        None: Control while the application serves requests.
    """
    settings = get_settings()
    app.state.settings = settings
    app.state.gemini = GeminiClient(settings)
    app.state.web_tools, web_clients = build_web_tools(settings)
    logger.info(
        "[APP] Starting up, web search %s",
        "enabled" if app.state.web_tools else "disabled",
    )
    yield
    for client in web_clients:
        await client.aclose()
    logger.info("[APP] Shutting down")


app = FastAPI(title="Chatbot", lifespan=lifespan)
app.include_router(health_router, tags=["Health"])
app.include_router(chat_router, prefix="/v1", tags=["Chat"])
