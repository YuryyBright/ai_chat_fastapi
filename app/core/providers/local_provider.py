# app/core/providers/local_universal_provider.py
"""
Універсальний локальний провайдер (працює з 95% всіх локальних бекендів 2025 року)
Підтримує:
- llama.cpp server (GGUF + всі кванти)
- tabbyAPI (EXL2 — найшвидше на NVIDIA)
- vLLM
- oobabooga text-generation-webui
- LM Studio
- KoboldCPP
- AnythingLLM
- Всі, хто має OpenAI-сумісний API
"""

import time
import httpx
from typing import AsyncGenerator, List, Dict, Any
from app.core.providers.base import BaseLLMProvider
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.core.exceptions import ProviderError
import logging

logger = logging.getLogger(__name__)


class LocalProvider(BaseLLMProvider):
    """
    Універсальний провайдер для будь-якого локального OpenAI-сумісного серверу
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://127.0.0.1:8080/v1")  # зазвичай так
        self.api_key = config.get("api_key", "not-needed")
        self.timeout = config.get("timeout", 600)
        self.client = None
        self.model_list_cache = None

    async def initialize(self) -> None:
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        logger.info(f"LocalUniversalProvider ініціалізовано: {self.base_url}")

    async def is_available(self) -> bool:
        try:
            resp = await self.client.get("/health")
            return resp.status_code in (200, 404)  # 404 теж ок — деякі сервери не мають /health
        except:
            try:
                await self.client.get("/v1/models")
                return True
            except:
                return False

    async def list_models(self) -> List[str]:
        if self.model_list_cache:
            return self.model_list_cache
        
        try:
            resp = await self.client.get("/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models = [m["id"] for m in data.get("data", [])]
                self.model_list_cache = models
                return models
        except Exception as e:
            logger.warning(f"Не вдалося отримати список моделей: {e}")
        
        return ["unknown-local-model"]

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_time = time.time()
        model = request.model or "default"

        messages = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        if request.context:
            messages.extend(request.context)
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": False
        }

        try:
            resp = await self.client.post("/v1/chat/completions", json=payload)
            if resp.status_code != 200:
                error = resp.text
                raise ProviderError(f"Local server error {resp.status_code}: {error}")

            data = resp.json()
            choice = data["choices"][0]
            generation_time = time.time() - start_time

            return GenerationResponse(
                generated_text=choice["message"]["content"],
                model=model,
                provider="local_universal",
                tokens_used=data.get("usage", {}).get("total_tokens"),
                finish_reason=choice.get("finish_reason"),
                generation_time=generation_time,
                metadata=data
            )

        except Exception as e:
            logger.error(f"LocalUniversal generate error: {e}")
            raise ProviderError(f"Локальний сервер не відповів: {str(e)}")

    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        model = request.model or "default"

        messages = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        if request.context:
            messages.extend(request.context)
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": True
        }

        try:
            async with self.client.stream("POST", "/v1/chat/completions", json=payload) as response:
                async for line in response.atext():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk.strip() == "[DONE]":
                            break
                        try:
                            data = httpx._models.Response.json({"content": chunk})
                            delta = data["choices"][0]["delta"].get("content")
                            if delta:
                                yield delta
                        except:
                            continue
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise ProviderError(f"Помилка стримінгу: {e}")

    async def cleanup(self) -> None:
        if self.client:
            await self.client.aclose()