"""Application entry point building the FastAPI app."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from chatbot.api.routes_chat import router as chat_router
from chatbot.api.routes_health import router as health_router
from chatbot.config import get_settings
from chatbot.core.llm.gemini import GeminiClient
from chatbot.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared resources on startup and log the application's lifetime.

    Args:
        app: The application whose state receives the shared resources.

    Yields:
        None: Control while the application serves requests.
    """
    app.state.gemini = GeminiClient(get_settings())
    logger.info("[APP] Starting up")
    yield
    logger.info("[APP] Shutting down")


app = FastAPI(title="Chatbot", lifespan=lifespan)
app.include_router(health_router, tags=["Health"])
app.include_router(chat_router, prefix="/v1", tags=["Chat"])
