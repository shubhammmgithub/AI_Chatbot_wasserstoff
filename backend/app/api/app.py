import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from backend.app.core.logger import setup_logger
from backend.app.core.config import API_TITLE, API_VERSION, CORS_ORIGINS
from backend.app.routes import api_router
from backend.app.utils.exceptions import WasserstoffException, get_error_response

# Loading the environment variables from .env file
load_dotenv()

# Setup logger
logger = setup_logger("api")

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Retrieval Augmented Generation (RAG) API for document search and question answering",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(api_router)


# Global exception handler for custom exceptions
@app.exception_handler(WasserstoffException)
async def wasserstoff_exception_handler(request, exc: WasserstoffException):
    error_response = get_error_response(exc)
    return JSONResponse(
        status_code=error_response.status_code,
        content={"detail": error_response.detail},
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {API_TITLE} v{API_VERSION}")