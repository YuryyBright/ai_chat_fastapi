
# app/api/routes/generation.py
"""
API роути для генерації тексту
Повністю сумісний з LocalProvider + SSE streaming
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator
import json
import logging

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

    # У тебе тільки local_provider → завжди використовуємо його
    provider = manager.get_provider("local_provider")

    logger.info(
        f"Генерація: model={request.model or 'авто'}, "
        f"prompt_len={len(request.prompt)}, temp={request.temperature}"
    )

    try:
        response = await provider.generate(request)
        return response
    except Exception as e:
        logger.error(f"Помилка генерації: {e}")
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
            logger.info(f"Starting generate stream")
            async for token in provider.generate_stream(request):
                logger.info("Генеруємо токен: {token}")
                if token.strip():  # відправляємо тільки непорожні токени
                    chunk = StreamChunk(text=token, done=False)
                    logger.info(f"Chunk: {chunk}")
                    yield f"data: {chunk.model_dump_json()}\n\n"

            # Кінець генерації
            final = StreamChunk(text="", done=True)
            yield f"data: {final.model_dump_json()}\n\n"

        except Exception as e:
            logger.error(f"Помилка в потоковій генерації: {e}", exc_info=True)
            error_chunk = StreamChunk(text="", done=True, error=str(e))
            yield f"data: {error_chunk.model_dump_json()}\n\n"

        # Додатковий порожній рядок — допомагає клієнтам
        yield "\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # важливо для Nginx
            "Access-Control-Allow-Origin": "*",
        },
    )
