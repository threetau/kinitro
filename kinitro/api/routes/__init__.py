"""API routes for the Kinitro evaluation backend."""

from kinitro.api.routes.health import router as health_router
from kinitro.api.routes.weights import router as weights_router
from kinitro.api.routes.scores import router as scores_router
from kinitro.api.routes.tasks import router as tasks_router
from kinitro.api.routes.miners import router as miners_router

__all__ = [
    "health_router",
    "weights_router",
    "scores_router",
    "tasks_router",
    "miners_router",
]
