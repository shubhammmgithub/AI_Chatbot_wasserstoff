from fastapi import HTTPException, status
from typing import Dict, Any, Optional


class WasserstoffException(Exception):
    """Base exception class for Wasserstoff AI application"""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class FileExtractionError(WasserstoffException):
    """Exception raised when file extraction fails"""
    def __init__(self, message: str, filename: Optional[str] = None):
        detail = f"Failed to extract content from file {filename}: {message}" if filename else message
        super().__init__(detail, status.HTTP_422_UNPROCESSABLE_ENTITY)


class ChunkingError(WasserstoffException):
    """Exception raised when text chunking fails"""
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_422_UNPROCESSABLE_ENTITY)


class EmbeddingError(WasserstoffException):
    """Exception raised when embedding generation fails"""
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)


class UpsertError(WasserstoffException):
    """Exception raised when upserting to vector database fails"""
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)


class RetrievalError(WasserstoffException):
    """Exception raised when retrieval from vector database fails"""
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)


class ModelGenerationError(WasserstoffException):
    """Exception raised when LLM generation fails"""
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)


class ConfigurationError(WasserstoffException):
    """Exception raised when configuration is invalid"""
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)


class ThemeExtractionError(WasserstoffException):
    """Exception raised when theme extraction fails"""
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)


def get_error_response(exception: Exception) -> Dict[str, Any]:
    """Convert exception to a standardized error response format"""
    if isinstance(exception, WasserstoffException):
        return {"detail": exception.message}
    elif isinstance(exception, HTTPException):
        return {"detail": exception.detail}
    else:
        return {"detail": str(exception)}