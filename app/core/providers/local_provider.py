# app/core/providers/local_provider.py
"""
ПОВНІСТЮ ОФЛАЙН локальний провайдер
Автоматично знаходить моделі в ./models:
- .gguf / .bin файли
- HuggingFace папки (з config.json тощо)
- EXL2 / GPTQ / AWQ (папки з .safetensors + config)
Працює через llama.cpp — найуніверсальнініше рішення 2025 року
"""
import os
from pathlib import Path
from typing import List, AsyncGenerator, Dict, Any
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.core.providers.base import BaseLLMProvider
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class LocalProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.models_dir = Path(settings.models_dir or "./models")
        self.loaded_models = {}  # кеш: model_name -> Llama instance

    async def initialize(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalProvider запущено. Шукаю моделі в: {self.models_dir}")

    async def is_available(self) -> bool:
        return True  # завжди доступний — офлайн

    async def list_models(self) -> List[str]:
        if not self.models_dir.exists():
            return []

        models = set()

        for path in self.models_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".gguf", ".bin", ".gptq", ".awq"}:
                models.add(path.stem)

            elif path.is_dir():
                # HuggingFace формат
                if any((path / f).exists() for f in ["config.json", "generation_config.json"]):
                    models.add(path.name)
                # EXL2, GPTQ, AWQ — шукаємо safetensors + config
                elif any(f.suffix == ".safetensors" for f in path.rglob("*.safetensors")):
                    if (path / "config.json").exists() or (path / "quantize_config.json").exists():
                        models.add(path.name)

        models_list = sorted(models)
        logger.info(f"Знайдено локальних моделей: {len(models_list)} → {models_list or 'немає'}")
        return models_list

    def _get_llama(self, model_name: str):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        model_path = self._resolve_path(model_name)
        if not model_path:
            raise ValueError(f"Модель '{model_name}' не знайдена в {self.models_dir}")

        try:
            from llama_cpp import Llama

            logger.info(f"Завантажую модель: {model_path}")
            llm = Llama(
                model_path=str(model_path),
                n_ctx=8192,
                n_batch=512,
                n_threads=max(1, os.cpu_count() - 1),
                n_gpu_layers=999 if os.getenv("USE_GPU", "0") == "1" else 0,
                verbose=False,
            )
            self.loaded_models[model_name] = llm
            return llm
        except Exception as e:
            logger.error(f"Не вдалося завантажити {model_path}: {e}")
            raise

    def _resolve_path(self, model_name: str) -> Path | None:
        candidates = [
            self.models_dir / f"{model_name}.gguf",
            self.models_dir / f"{model_name}.bin",
            self.models_dir / model_name,  # папка HF / EXL2
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        result = ""
        async for token in self.generate_stream(request):
            result += token
        return GenerationResponse(generated_text=result)

    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        model_name = request.model or next(iter(await self.list_models()), "unknown")
        llm = self._get_llama(model_name)

        stream = llm(
            prompt=request.prompt,
            max_tokens=request.max_tokens or 1024,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
            stop=request.stop_sequences,
            stream=True,
        )

        for chunk in stream:
            text = chunk["choices"][0]["delta"].get("content", "")
            if text:
                yield text