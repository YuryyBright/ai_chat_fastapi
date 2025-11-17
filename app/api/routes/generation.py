# app/api/routes/generation.py
"""
API routes for text generation
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.core.providers import ProviderManager
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def get_provider_manager(request: Request) -> ProviderManager:
    """Dependency to get provider manager from app state"""
    return request.app.state.provider_manager


@router.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    app_request: Request
) -> GenerationResponse:
    """
    Generate text using specified provider and model
    
    - **prompt**: Input text prompt
    - **provider**: LLM provider (ollama, huggingface, openai)
    - **model**: Model name (optional, uses default if not specified)
    - **stream**: Enable streaming (use /generate/stream endpoint instead)
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="For streaming generation, use the /generate/stream endpoint"
        )
    
    manager = get_provider_manager(app_request)
    provider = manager.get_provider(request.provider)
    
    logger.info(
        f"Generation request: provider={request.provider}, "
        f"model={request.model}, prompt_length={len(request.prompt)}"
    )
    
    response = await provider.generate(request)
    return response


@router.post("/generate/stream")
async def generate_text_stream(
    request: GenerationRequest,
    app_request: Request
):
    """
    Generate text with streaming output
    
    Returns Server-Sent Events (SSE) stream of generated text chunks
    """
    manager = get_provider_manager(app_request)
    provider = manager.get_provider(request.provider)
    
    logger.info(
        f"Streaming generation request: provider={request.provider}, "
        f"model={request.model}"
    )
    
    async def event_generator():
        """Generate SSE events"""
        try:
            async for chunk in provider.generate_stream(request):
                # Format as SSE
                data = json.dumps({"text": chunk, "done": False})
                yield f"data: {data}\n\n"
            
            # Send completion event
            data = json.dumps({"text": "", "done": True})
            yield f"data: {data}\n\n"
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_data = json.dumps({"error": str(e), "done": True})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )