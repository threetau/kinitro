from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Match, Route

from backend.auth import UserRole, get_api_key_from_db


ADMIN_FLAG_ATTR = "__requires_admin__"


def admin_route(endpoint: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Mark an endpoint as requiring an authenticated admin."""
    setattr(endpoint, ADMIN_FLAG_ATTR, True)
    return endpoint


class ApiAuthMiddleware(BaseHTTPMiddleware):
    """Attach authenticated API user to request state when a valid key is provided."""

    def __init__(self, app, backend_service):
        super().__init__(app)
        self.backend_service = backend_service

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Any]]):
        request.state.api_user = None
        api_key = request.headers.get("X-API-Key")

        if api_key:
            api_user = await get_api_key_from_db(api_key, self.backend_service)
            if not api_user:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid, expired, or inactive API key"},
                )
            request.state.api_user = api_user

        return await call_next(request)


class AdminAuthMiddleware(BaseHTTPMiddleware):
    """Enforce admin-only access for endpoints marked with admin_route."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Any]]):
        route = self._get_matching_route(request)

        if route and getattr(route.endpoint, ADMIN_FLAG_ATTR, False):
            api_user = getattr(request.state, "api_user", None)

            if api_user is None:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authentication required"},
                )

            if api_user.role != UserRole.ADMIN:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Admin access required"},
                )

        return await call_next(request)

    @staticmethod
    def _get_matching_route(request: Request) -> Optional[Route]:
        for route in request.app.router.routes:
            if not hasattr(route, "endpoint"):
                continue

            match, _ = route.matches(request.scope)
            if match == Match.FULL:
                return route  # type: ignore[return-value]
        return None
