"""Correlation ID middleware for the FastAPI gateway."""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.shared.logging import correlation_id_var


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Generate a unique correlation ID per request and propagate it."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        token = correlation_id_var.set(cid)
        try:
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = cid
            return response
        finally:
            correlation_id_var.reset(token)
