import os
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
import google.generativeai as genai

from backend.app.core.logger import setup_logger

# Setup logger
logger = setup_logger("config")

# Load environment variables
load_dotenv()

# --- Qdrant Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
QDRANT_THEMES_COLLECTION = os.getenv("QDRANT_THEMES_COLLECTION", "document_themes")

# --- Model Configuration ---
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
FALLBACK_DIM = 384  # Dimension for all-MiniLM-L6-v2

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b") # Final answer model
GEMINI_RERANK_MODEL = os.getenv("GEMINI_RERANK_MODEL", "gemini-1.5-flash-latest") # Reranking model
GEMINI_SMALL_MODEL = os.getenv("GEMINI_SMALL_MODEL", "gemini-1.5-flash") # Optional small model

# --- OCR Configuration ---
OCR_THRESHOLD = int(os.getenv("OCR_THRESHOLD", "150"))
OCR_LANG = os.getenv("OCR_LANG", "eng")
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
TESSERACT_CONFIG = "--oem 3 --psm 6"

# --- API Configuration ---
API_TITLE = "Wasserstoff AI Q/A & Theme Chatbot"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Document research + theme identification chatbot (RAG + semantic themes)."

# --- CORS Configuration ---
default_origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:3000",  # React dev
    "http://127.0.0.1:3000",
    "http://localhost:5173",  # Vite dev (optional)
    "http://127.0.0.1:5173",
    "http://localhost:8501",  # Streamlit
    "http://127.0.0.1:8501",
]
env_origins = os.getenv("FRONTEND_ORIGINS", "")
extra = [o.strip() for o in env_origins.split(",") if o.strip()]
CORS_ORIGINS = sorted(set(default_origins + extra))


# --- Client Initializers and Config Checks ---

def get_qdrant_client() -> QdrantClient:
    """Initialize and return the Qdrant client."""
    logger.debug(f"Initializing Qdrant client with URL: {QDRANT_URL}")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Backward-compatible alias
def qdrant_client() -> QdrantClient:
    return get_qdrant_client()


#This function is no longer used.
# because we upgraded our application
# to use the LangChain framework.

# def configure_genai():
#     """Configure and return the Gemini client module."""
#     if not GROQ_API_KEY:
#         raise ValueError("GROQ_API_KEY is not set.")
#     genai.configure(api_key=GROQ_API_KEY)
#     return genai


def assert_config():
    """Basic sanity checks for required configuration."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY must be set in the environment.")
        raise ValueError("GROQ_API_KEY must be set in the environment.")
    if not QDRANT_URL:
        logger.error("QDRANT_URL must be set.")
        raise ValueError("QDRANT_URL must be set.")
    
    logger.info("Configuration validation passed")