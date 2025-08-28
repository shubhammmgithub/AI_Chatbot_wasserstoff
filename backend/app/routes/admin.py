from fastapi import APIRouter, HTTPException, Header, Depends
from typing import Dict, Any

from backend.app.core.logger import setup_logger
from backend.app.core.config import get_qdrant_client

# Setup logger
logger = setup_logger("admin_routes")

# Create router with admin tag
router = APIRouter(prefix="/admin", tags=["admin"])


@router.post(
    "/session/end",
    summary="End a session and delete its data",
    response_model=Dict[str, str]
)
async def end_session(
    session_id: str = Header(..., alias="X-Session-ID"),
    qdrant_client = Depends(get_qdrant_client)
) -> Dict[str, str]:
    """
    Deletes the Qdrant collection associated with a specific session ID,
    effectively clearing all data for that user's session.
    """
    # Construct the unique collection name from the session ID
    collection_name = f"session_{session_id}"
    logger.info(f"Received request to end session, deleting collection: {collection_name}")
    
    try:
        # Delete the specific collection for this session
        qdrant_client.delete_collection(collection_name=collection_name)
        logger.info(f"Successfully deleted collection '{collection_name}' for session '{session_id}'.")
        return {"status": "success", "message": f"Session data for {session_id} has been deleted."}
    except Exception as e:
        # It's not a critical error if the collection didn't exist in the first place.
        # This can happen if the user clicks the button twice.
        logger.warning(f"Could not delete collection '{collection_name}' (it may have already been deleted): {e}")
        return {"status": "success", "message": "Session ended or data was already cleared."}