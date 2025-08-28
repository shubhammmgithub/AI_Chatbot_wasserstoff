from fastapi import APIRouter, Depends
from typing import Dict, Any

from backend.app.core.config import (
    QDRANT_COLLECTION,
    QDRANT_THEMES_COLLECTION,
    GROK_MODEL,
    GEMINI_RERANK_MODEL,
)
from backend.app.core.logger import setup_logger
from backend.app.models.schemas import HealthResponse

# Setup logger
logger = setup_logger("health_routes")

# Create router
router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify the API is running and configured correctly.
    
    Returns:
        Dict with status and configuration information
    """
    logger.info("Health check requested")
    return {
        "status": "ok",
        "models": {
            "answer_model": GROK_MODEL,
            "rerank_model": GEMINI_RERANK_MODEL,
        },
        "collections": {
            "primary": QDRANT_COLLECTION,
            "themes": QDRANT_THEMES_COLLECTION,
        },
    }