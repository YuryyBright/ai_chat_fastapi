# app/main.py
"""
FastAPI LLM Service - Main Application Entry Point

This service provides a unified interface for multiple LLM providers including:
- Ollama (local models)
- HuggingFace models
- OpenAI GPT models
- Custom fine-tuned models

Features:
- Streaming and non-streaming text generation
- Model fine-tuning capabilities
- Docker support
- Comprehensive error handling
- Request validation with Pydantic
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from app.config import settings
from app.api.routes import generation, models, training
from app.core.providers import ProviderManager
from app.core.exceptions import setup_exception_handlers

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan context manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting LLM Service...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Enabled providers: {settings.enabled_providers}")
    
    # Initialize provider manager
    app.state.provider_manager = ProviderManager()
    await app.state.provider_manager.initialize()
    
    logger.info("LLM Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Service...")
    await app.state.provider_manager.cleanup()
    logger.info("LLM Service stopped")


# Initialize FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Unified API for multiple LLM providers",
    version=settings.version,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup exception handlers
setup_exception_handlers(app)

# Include routers
app.include_router(
    generation.router,
    prefix="/api/v1/generation",
    tags=["Text Generation"]
)
app.include_router(
    models.router,
    prefix="/api/v1/models",
    tags=["Model Management"]
)
app.include_router(
    training.router,
    prefix="/api/v1/training",
    tags=["Model Training"]
)


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "service": settings.app_name,
        "version": settings.version,
        "status": "operational",
        "environment": settings.environment
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "providers": {
            provider: "available" 
            for provider in settings.enabled_providers
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development"
    )