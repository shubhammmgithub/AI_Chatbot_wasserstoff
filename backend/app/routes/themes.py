import json
import asyncio
from fastapi import APIRouter, Depends, Header
from fastapi.responses import StreamingResponse
from typing import Dict, Any

from backend.app.core.logger import setup_logger
from backend.app.services.theme_service import ThemeService

# Setup logger
logger = setup_logger("themes_routes")

# Create router
router = APIRouter(prefix="/themes", tags=["themes"])

def get_theme_service():
    """Dependency injector for the ThemeService."""
    return ThemeService()

@router.get(
    "/count",
    summary="Count unique themes for a session",
    response_model=int
)
async def count_themes(
    session_id: str = Header(..., alias="X-Session-ID"),
    theme_service: ThemeService = Depends(get_theme_service)
) -> int:
    """
    Returns the total number of unique themes for the user's current session.
    """
    logger.info(f"Counting themes for session_id: {session_id}")
    return theme_service.count_unique_themes(session_id=session_id)

@router.post(
    "/analyze-stream",
    summary="Analyze all themes for a session and stream results"
)
async def analyze_themes_stream(
    session_id: str = Header(..., alias="X-Session-ID"),
    theme_service: ThemeService = Depends(get_theme_service)
):
    """
    Performs a full analysis on all unique themes within a user's session
    and streams the result for each theme as it's completed.
    """
    logger.info(f"Starting theme analysis stream for session_id: {session_id}")
    async def stream_generator():
        try:
            for result in theme_service.analyze_all_themes_stream(session_id=session_id):
                yield f"data: {json.dumps(result)}\n\n"
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.exception(f"Streaming generator failed for session {session_id}: {e}")
            error_message = {"error": "A critical error occurred in the streaming generator."}
            yield f"data: {json.dumps(error_message)}\n\n"
    
    return StreamingResponse(stream_generator(), media_type="text/event-stream")