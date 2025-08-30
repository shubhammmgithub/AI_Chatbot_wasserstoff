from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# --- Request Models ---
class AskRequest(BaseModel):
    """Request model for querying the knowledge base"""
    query: str = Field(..., min_length=1, description="The query text to search for")
    top_k: int = Field(20, description="Number of initial results to retrieve")
    final_n: int = Field(5, description="Number of final results after reranking")



class IngestRequest(BaseModel):
    """Request model for ingestion parameters"""
    chunk_size: int = Field(1024, description="Size of text chunks")
    overlap: int = Field(200, description="Overlap between chunks")
    batch_size: int = Field(64, description="Batch size for upserting")
    enable_themes: bool = Field(True, description="Whether to enable theme extraction")


class ThemeRequest(BaseModel):
    """Request model for theme operations"""
    limit: int = Field(10, description="Maximum number of themes to return")
    offset: int = Field(0, description="Offset for pagination")
    doc_id: Optional[str] = Field(None, description="Filter by document ID")


# --- Response Models ---
class SupportingChunk(BaseModel):
    """Model for a supporting chunk in the response"""
    rank: int
    rerank_score: float
    doc_id: str
    page: Optional[int] = None
    para: Optional[int] = None
    theme: Optional[str] = None
    text: str
    vector_score: Optional[float] = None
    id: Optional[str] = None


class AskResponse(BaseModel):
    """Response model for query results"""
    answer: str
    supporting_chunks: List[SupportingChunk]


class Citation(BaseModel):
    """Model for a citation in theme responses"""
    doc_id: str
    page: Optional[int] = None
    para: Optional[int] = None
    snippet: Optional[str] = None


class Theme(BaseModel):
    """Model for a theme"""
    id: Optional[str] = None
    label: str
    citations: List[Citation]


class ThemeResponse(BaseModel):
    """Response model for theme results"""
    items: List[Theme]
    next_offset: Optional[int] = None


class IngestResponse(BaseModel):
    """Response model for ingestion results"""
    filename: str
    chunks_extracted: int
    #chunks_upserted: int


class BatchIngestResponse(BaseModel):
    """Response model for batch ingestion results"""
    results: List[IngestResponse]
    total_chunks_extracted: int
    total_chunks_upserted: int


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    models: Dict[str, str]
    collections: Dict[str, str]
    

class ChatMessage(BaseModel):
    """Model for a chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    
 
class AskRequest(BaseModel):
    """Request model for querying the knowledge base with chat history"""
    query: str = Field(..., min_length=1)
    top_k: int = Field(20)
    final_n: int = Field(5)
    chat_history: Optional[List[ChatMessage]] = None


class ErrorResponse(BaseModel):
    """Response model for errors"""
    detail: str