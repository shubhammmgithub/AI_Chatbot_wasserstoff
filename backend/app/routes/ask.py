from fastapi import APIRouter, Query, Depends, HTTPException, Header
from typing import Optional

from backend.app.core.logger import setup_logger
from backend.app.models.schemas import AskRequest, AskResponse, ErrorResponse
from backend.app.services.retrieval_service import RetrievalService

# Setup logger
logger = setup_logger("ask_routes")

# Create router
router = APIRouter(prefix="/ask", tags=["ask"])

# Dependency for services
def get_retrieval_service():
    return RetrievalService()


@router.get("", response_model=AskResponse)
async def ask_question_get(
    q: str = Query(..., min_length=1, description="User query"),
    session_id: str = Header(..., alias="X-Session-ID"),
    top_k: int = Query(20, description="Number of initial results to retrieve"),
    final_n: int = Query(5, description="Number of final results after reranking"),
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
) -> AskResponse:
    """Asks a question to a user's private, session-based knowledge base using GET."""
    logger.info(f"Processing query (GET) for session_id: {session_id}")
    try:
        result = retrieval_service.retrieve_and_answer(
            query=q,
            session_id=session_id,
            top_k=top_k,
            final_n=final_n
        )
        return AskResponse(**result)
    except Exception as e:
        logger.exception(f"Unexpected error during GET retrieval for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}")


@router.post("", response_model=AskResponse)
async def ask_question_post(
    body: AskRequest,
    session_id: str = Header(..., alias="X-Session-ID"),
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
) -> AskResponse:
    """Asks a question to a user's private, session-based knowledge base using POST."""
    logger.info(f"Processing query (POST) for session_id: {session_id}")
    try:
        result = retrieval_service.retrieve_and_answer(
            query=body.query,
            session_id=session_id,
            top_k=body.top_k,
            final_n=body.final_n,
            chat_history=[msg.dict() for msg in body.chat_history] if body.chat_history else None  
        )
        return AskResponse(**result)
    except Exception as e:
        logger.exception(f"Unexpected error during POST retrieval for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}")