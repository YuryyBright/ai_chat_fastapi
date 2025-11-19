# app/api/routes/generation.py
"""
API роути для генерації тексту
Виправлені проблеми зі стрімінгом та обробкою відповідей
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator
import json
import logging
import asyncio

from app.schemas.generation import GenerationRequest, GenerationResponse
from app.core.providers import ProviderManager

logger = logging.getLogger(__name__)
router = APIRouter()


def get_provider_manager(request: Request) -> ProviderManager:
    """Залежність: отримуємо менеджер провайдерів з app.state"""
    if not hasattr(request.app.state, "provider_manager"):
        raise HTTPException(status_code=503, detail="Provider manager not initialized")
    return request.app.state.provider_manager


class StreamChunk(BaseModel):
    """Модель для SSE чанку"""
    text: str = ""
    done: bool = False
    error: str | None = None


@router.post("", response_model=GenerationResponse)
@router.post("/", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    manager: ProviderManager = Depends(get_provider_manager),
) -> GenerationResponse:
    """
    Звичайна (не потокова) генерація тексту
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Для потокової генерації використовуйте POST /generate/stream"
        )

    provider = manager.get_provider("local_provider")

    logger.info(
        f"Генерація: model={request.model or 'авто'}, "
        f"prompt_len={len(request.prompt)}, temp={request.temperature}"
    )

    try:
        response = await provider.generate(request)
        logger.info(f"✓ Згенеровано {len(response.generated_text)} символів")
        return response
    except Exception as e:
        logger.error(f"✗ Помилка генерації: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def generate_text_stream(
    request: GenerationRequest,
    manager: ProviderManager = Depends(get_provider_manager),
):
    """
    Потокова генерація через Server-Sent Events (SSE)
    
    Формат чанку:
    data: {"text": "Привіт", "done": false}
    data: {"text": "", "done": true}
    """
    provider = manager.get_provider("local_provider")

    logger.info(
        f"Потокова генерація: model={request.model or 'авто'}, "
        f"prompt_len={len(request.prompt)}"
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            token_count = 0
            last_yield_time = asyncio.get_event_loop().time()
            
            # Відправляємо початковий heartbeat
            yield ": keepalive\n\n"
            
            async for token in provider.generate_stream(request):
                current_time = asyncio.get_event_loop().time()
                
                # Heartbeat кожні 15 секунд для підтримки з'єднання
                if current_time - last_yield_time > 15:
                    yield ": keepalive\n\n"
                    last_yield_time = current_time
                
                if token:  # відправляємо будь-який непорожній текст
                    chunk = StreamChunk(text=token, done=False)
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    token_count += 1
                    last_yield_time = current_time
                    
                    # Flush кожні 5 токенів для кращого UX
                    if token_count % 5 == 0:
                        await asyncio.sleep(0)  # Дозволяємо іншим задачам виконатися

            # Кінець генерації
            logger.info(f"✓ Згенеровано {token_count} токенів")
            final = StreamChunk(text="", done=True)
            yield f"data: {final.model_dump_json()}\n\n"

        except asyncio.CancelledError:
            logger.info("Потік скасовано клієнтом")
            error_chunk = StreamChunk(text="", done=True, error="Скасовано")
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            
        except Exception as e:
            logger.error(f"✗ Помилка в потоковій генерації: {e}", exc_info=True)
            error_chunk = StreamChunk(text="", done=True, error=str(e))
            yield f"data: {error_chunk.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # важливо для Nginx
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )