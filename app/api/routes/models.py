# app/api/routes/models.py
"""
API routes for model management
"""

from fastapi import APIRouter, Request
from app.schemas.models import ModelInfo, ModelListResponse
from typing import List
import logging
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download
import json
import os
from pathlib import Path
from pydantic import BaseModel
logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/list", response_model=ModelListResponse)
async def list_all_models(request: Request) -> ModelListResponse:
    """
    List all available models from all providers
    
    Returns a comprehensive list of models grouped by provider
    """
    manager = request.app.state.provider_manager
    all_models = await manager.get_all_models()
    
    models_list = []
    for provider, model_names in all_models.items():
        for model_name in model_names:
            models_list.append(
                ModelInfo(
                    name=model_name,
                    provider=provider,
                    type="language_model",
                    capabilities=["text_generation"],
                    status="available"
                )
            )
    
    logger.info(f"Listed {len(models_list)} models across {len(all_models)} providers")
    
    return ModelListResponse(
        models=models_list,
        total=len(models_list)
    )


@router.get("/list/{provider}", response_model=List[str])
async def list_provider_models(
    provider: str,
    request: Request
) -> List[str]:
    """
    List available models for a specific provider
    
    - **provider**: Provider name (ollama, huggingface, openai)
    """
    manager = request.app.state.provider_manager
    provider_instance = manager.get_provider(provider)
    
    models = await provider_instance.list_models()
    logger.info(f"Listed {len(models)} models for provider '{provider}'")
    
    return models


@router.get("/health")
async def check_providers_health(request: Request):
    """
    Check health status of all providers
    
    Returns availability status for each enabled provider
    """
    manager = request.app.state.provider_manager
    health_status = await manager.health_check()
    
    return {
        "status": "healthy" if all(health_status.values()) else "degraded",
        "providers": health_status
    }

# ← Правильна модель для JSON-body
class PullModelRequest(BaseModel):
    model_name: str
    provider: str = "huggingface"

async def progress_generator(repo_id: str, cache_dir: str):
    # Спочатку надсилаємо "початок"
    yield f"data: {json.dumps({'status': 'progress', 'progress': 0.0})}\n\n"

    try:
        # Просто запускаємо завантаження — huggingface_hub сам виводить tqdm у консоль сервера
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False,
        )
        yield f"data: {json.dumps({'status': 'complete', 'model': repo_id})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"


@router.post("/pull")
async def pull_model(request: PullModelRequest):
    if request.provider != "huggingface":
        return {"error": "Only huggingface provider supported"}

    cache_dir = os.getenv("HUGGINGFACE_CACHE_DIR", "./models/huggingface")

    return StreamingResponse(
        progress_generator(request.model_name, cache_dir),
        media_type="text/event-stream",
    )