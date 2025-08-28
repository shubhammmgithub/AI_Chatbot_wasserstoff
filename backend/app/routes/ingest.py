import os
import tempfile
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Header

from backend.app.core.logger import setup_logger
from backend.app.models.schemas import IngestResponse, BatchIngestResponse
from backend.app.services.extraction_service import ExtractionService
from backend.app.services.embedding_service import EmbeddingService

# Setup logger
logger = setup_logger("ingest_routes")
router = APIRouter(prefix="/ingest", tags=["ingest"])

def get_extraction_service():
    return ExtractionService()

def get_embedding_service():
    return EmbeddingService()

def _save_upload_to_temp(upload: UploadFile) -> str:
    """Saves an uploaded file to a temporary location on disk."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload.filename)[1]) as tmp:
            tmp.write(upload.file.read())
            return tmp.name
    except Exception as e:
        logger.exception(f"Failed to save upload {upload.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {upload.filename}")

@router.post("/batch", response_model=BatchIngestResponse)
async def ingest_batch(
    files: List[UploadFile] = File(...),
    session_id: str = Header(..., alias="X-Session-ID"),
    extraction_service: ExtractionService = Depends(get_extraction_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> BatchIngestResponse:
    
    logger.info(f"Starting batch ingest for session_id: {session_id} with {len(files)} files.")
    
    results = []
    all_chunks = []

    # 1. Extract chunks from all files first
    for f in files:
        tmp_path = ""
        try:
            tmp_path = _save_upload_to_temp(f)
            chunks = extraction_service.extract_and_chunk(tmp_path)
            if chunks:
                all_chunks.extend(chunks)
            results.append({"filename": f.filename, "chunks_extracted": len(chunks)})
        except Exception as e:
            logger.error(f"Failed during extraction for file {f.filename}: {e}")
            results.append({"filename": f.filename, "chunks_extracted": 0})
        finally:
            if tmp_path:
                os.unlink(tmp_path)

    # 2. Embed and upsert all chunks in a single operation
    total_chunks_extracted = sum(r['chunks_extracted'] for r in results)
    total_chunks_upserted = 0
    if all_chunks:
        try:
            logger.info(f"Upserting {len(all_chunks)} total chunks for session {session_id}.")
            total_chunks_upserted = embedding_service.upsert_chunks(
                chunks=all_chunks, 
                session_id=session_id
            )
        except Exception as e:
            logger.error(f"Failed during upserting for session {session_id}: {e}")

    return BatchIngestResponse(
        results=results,
        total_chunks_extracted=total_chunks_extracted,
        total_chunks_upserted=total_chunks_upserted,
    )