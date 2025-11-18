# app/api/routes/models.py
"""
API роути для керування моделями
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import logging

from app.schemas.models import ModelInfo, ModelListResponse
from app.core.model_downloader import ModelDownloader

logger = logging.getLogger(__name__)
router = APIRouter()


class DownloadModelRequest(BaseModel):
    """Запит на завантаження моделі"""
    repo_id: str
    model_type: str = "gguf"  # gguf, huggingface, gptq, awq
    filename: Optional[str] = None  # Для GGUF - конкретний файл


class SearchModelsRequest(BaseModel):
    """Запит на пошук моделей"""
    query: str = ""
    limit: int = 20


class DeleteModelRequest(BaseModel):
    """Запит на видалення моделі"""
    model_name: str


@router.get("/list", response_model=ModelListResponse)
async def list_all_models(request: Request) -> ModelListResponse:
    """
    Список всіх локальних моделей
    """
    manager = request.app.state.provider_manager
    provider = manager.get_provider("local_provider")
    
    model_names = await provider.list_models() or []
    
    models_list = []
    for model_name in model_names:
        info = provider.get_model_info(model_name)
        
        models_list.append(
            ModelInfo(
                name=model_name,
                provider="local_provider",
                type="language_model",
                capabilities=["text_generation", "streaming"],
                status="available" if info else "unknown",
                metadata=info
            )
        )
    
    logger.info(f"Знайдено {len(models_list)} локальних моделей")
    
    return ModelListResponse(
        models=models_list,
        total=len(models_list)
    )


@router.get("/health")
async def check_provider_health(request: Request):
    """
    Перевірка стану провайдера
    """
    manager = request.app.state.provider_manager
    
    try:
        provider = manager.get_provider("local_provider")
        is_available = await provider.is_available()
        
        return {
            "status": "healthy" if is_available else "unavailable",
            "provider": "local_provider",
            "available": is_available
        }
    except Exception as e:
        logger.error(f"Помилка перевірки здоров'я: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/download")
async def download_model(
    request_data: DownloadModelRequest,
    request: Request
):
    """
    Завантаження моделі з HuggingFace
    
    Повертає Server-Sent Events з прогресом завантаження
    """
    downloader: ModelDownloader = request.app.state.model_downloader
    
    async def event_generator():
        try:
            async for progress in downloader.download_model(
                repo_id=request_data.repo_id,
                model_type=request_data.model_type,
                filename=request_data.filename
            ):
                yield f"data: {json.dumps(progress)}\n\n"
        except Exception as e:
            logger.error(f"Помилка завантаження: {e}")
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@router.post("/search")
async def search_models(
    search_request: SearchModelsRequest,
    request: Request
):
    """
    Пошук доступних моделей на HuggingFace
    """
    downloader: ModelDownloader = request.app.state.model_downloader
    
    try:
        models = await downloader.list_available_models(
            query=search_request.query
        )
        
        return {
            "status": "success",
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"Помилка пошуку: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{repo_id:path}")
async def list_model_files(
    repo_id: str,
    request: Request
):
    """
    Отримати список файлів в репозиторії HuggingFace
    """
    downloader: ModelDownloader = request.app.state.model_downloader
    
    try:
        files = downloader.get_model_files(repo_id)
        
        return {
            "status": "success",
            "repo_id": repo_id,
            "files": files,
            "count": len(files)
        }
    except Exception as e:
        logger.error(f"Помилка отримання файлів: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete")
async def delete_model(
    delete_request: DeleteModelRequest,
    request: Request
):
    """
    Видалення локальної моделі
    """
    downloader: ModelDownloader = request.app.state.model_downloader
    
    try:
        success = await downloader.delete_model(delete_request.model_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Модель {delete_request.model_name} успішно видалена"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Модель {delete_request.model_name} не знайдена"
            )
    except Exception as e:
        logger.error(f"Помилка видалення: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info/{model_name:path}")
async def get_model_info(
    model_name: str,
    request: Request
):
    """
    Отримати детальну інформацію про модель
    """
    manager = request.app.state.provider_manager
    provider = manager.get_provider("local_provider")
    downloader: ModelDownloader = request.app.state.model_downloader
    
    try:
        info = provider.get_model_info(model_name)
        
        if not info:
            raise HTTPException(
                status_code=404,
                detail=f"Модель {model_name} не знайдена"
            )
        
        # Додаємо інформацію про розмір
        size_bytes = downloader.get_model_size(model_name)
        info["size"] = downloader.format_size(size_bytes)
        info["size_bytes"] = size_bytes
        
        return {
            "status": "success",
            "model": info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Помилка отримання інформації: {e}")
        raise HTTPException(status_code=500, detail=str(e))