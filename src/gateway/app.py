"""FastAPI gateway application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.shared.logging import setup_logging, get_logger
from src.shared.settings import Settings
from src.gateway.middleware import CorrelationIdMiddleware
from src.gateway.routes.health import router as health_router
from src.gateway.routes.chat import router as chat_router
from src.gateway.routes.index import router as index_router
from src.gateway.routes.graph import router as graph_router

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle hook."""
    setup_logging("gateway", settings.LOG_LEVEL)
    logger = get_logger("gateway")
    logger.info("Gateway starting up")
    yield
    logger.info("Gateway shutting down")


app = FastAPI(
    title="FastAPI Repo Chat Agent",
    version="0.1.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(CorrelationIdMiddleware)

# Routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(index_router)
app.include_router(graph_router)
