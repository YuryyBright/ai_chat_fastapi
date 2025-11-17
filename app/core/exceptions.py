# app/core/exceptions.py
"""
Custom exceptions for the application
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)


class LLMServiceException(Exception):
    """Base exception for LLM service"""
    
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ProviderError(LLMServiceException):
    """Exception raised when provider operations fail"""
    
    def __init__(self, message: str):
        super().__init__(message, status_code=503)


class ModelNotFoundError(LLMServiceException):
    """Exception raised when model is not found"""
    
    def __init__(self, message: str):
        super().__init__(message, status_code=404)


class TrainingError(LLMServiceException):
    """Exception raised during model training"""
    
    def __init__(self, message: str):
        super().__init__(message, status_code=500)


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup exception handlers for the FastAPI application"""
    
    @app.exception_handler(LLMServiceException)
    async def llm_service_exception_handler(
        request: Request,
        exc: LLMServiceException
    ):
        """Handle custom LLM service exceptions"""
        logger.error(f"LLM Service Error: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "message": exc.message,
                "path": str(request.url)
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ):
        """Handle request validation errors"""
        logger.warning(f"Validation Error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "ValidationError",
                "message": "Request validation failed",
                "details": exc.errors()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception
    ):
        """Handle unexpected exceptions"""
        logger.error(f"Unexpected Error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "details": str(exc) if settings.environment == "development" else None
            }
        )
