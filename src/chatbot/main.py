"""Application entry point building the FastAPI app."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from chatbot.api.routes_health import router as health_router
from chatbot.utils.logger import logger


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Run startup and shutdown work around the application's lifetime.

    Yields:
        None: Control while the application serves requests.
    """
    logger.info("[APP] Starting up")
    yield
    logger.info("[APP] Shutting down")


app = FastAPI(title="Chatbot", lifespan=lifespan)
app.include_router(health_router, tags=["Health"])
