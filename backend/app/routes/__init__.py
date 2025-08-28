import os
from fastapi import APIRouter

from backend.app.routes.health import router as health_router
from backend.app.routes.ingest import router as ingest_router
from backend.app.routes.ask import router as ask_router
from backend.app.routes.themes import router as themes_router
from backend.app.routes.admin import router as admin_router

# Create main API router
api_router = APIRouter()

# Include all routers
api_router.include_router(health_router)
api_router.include_router(ingest_router)
api_router.include_router(ask_router)
api_router.include_router(themes_router)
#api_router.include_router(admin_router)

if os.getenv("ENVIRONMENT") == "development":
    api_router.include_router(admin_router)
