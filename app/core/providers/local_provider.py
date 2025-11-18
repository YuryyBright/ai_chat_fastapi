# app/core/providers/local_provider.py
import os
from pathlib import Path
from typing import List, AsyncGenerator, Dict, Any, Optional
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.core.providers.base import BaseLLMProvider
from app.config import settings
import logging
import time
from datetime import datetime
logger = logging.getLogger(__name__)

class LocalProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.models_dir = Path(settings.models_dir or "./models")
        self.loaded_models = {}  # model_name -> Llama instance

    async def initialize(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalProvider ініціалізовано. Папка моделей: {self.models_dir.resolve()}")

    async def is_available(self) -> bool:
        return True

    async def list_models(self) -> List[str]:
        if not self.models_dir.exists():
            return []

        models = set()
        for path in self.models_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".gguf", ".bin"}:
                models.add(path.stem)
            elif path.is_dir():
                if (path / "config.json").exists() or any(f.suffix == ".safetensors" for f in path.glob("*.safetensors")):
                    models.add(path.name)

        models_list = sorted(models)
        logger.info(f"Знайдено локальних моделей: {len(models_list)}")
        return models_list

    def _resolve_path(self, model_name: str) -> Optional[Path]:
        candidates = [
            self.models_dir / f"{model_name}.gguf",
            self.models_dir / f"{model_name}.bin",
            self.models_dir / model_name,
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _get_llama(self, model_name: str):
        """Завантаження моделі з правильними параметрами"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        model_path = self._resolve_path(model_name)
        if not model_path:
            raise ValueError(f"Модель '{model_name}' не знайдена в {self.models_dir}")

        try:
            from llama_cpp import Llama

            n_gpu_layers = self.config.get("gpu_layers", -1)
            if n_gpu_layers == -1:
                n_gpu_layers = 999  # усі шари на GPU

            # КРИТИЧНО: logits_all=False для швидкості
            llm = Llama(
                model_path=str(model_path),
                n_ctx=settings.context_size or 8192,
                n_batch=512,
                n_threads=settings.n_threads or max(1, os.cpu_count() - 1),
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                logits_all=False,  # ОБОВ'ЯЗКОВО False!
                use_mlock=True,
                use_mmap=True,
            )
            
            self.loaded_models[model_name] = llm
            logger.info(f"✓ Завантажено модель: {model_name} (GPU layers: {n_gpu_layers})")
            return llm
            
        except Exception as e:
            logger.error(f"✗ Помилка завантаження моделі {model_name}: {e}")
            raise

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_time = time.time()
        result = ""

        async for token in self.generate_stream(request):
            result += token

        generation_time = time.time() - start_time

        available_models = await self.list_models()
        model_name = request.model or (available_models[0] if available_models else "unknown")

        return GenerationResponse(
            generated_text=result.strip(),
            model=model_name,
            provider="local",
            generation_time=round(generation_time, 3),
        )

    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Потокова генерація з правильними параметрами"""
        available_models = await self.list_models()
        model_name = request.model or (available_models[0] if available_models else None)
        
        if not model_name:
            raise ValueError("Немає доступних моделей")

        llm = self._get_llama(model_name)

        # Параметри генерації (БЕЗ cache_prompt тут!)
        completion_kwargs = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature or 0.7,
            "top_p": request.top_p or 0.9,
            "top_k": request.top_k or 40,
            "repeat_penalty": 1.1,
            "stop": request.stop_sequences or [],
            "stream": True,
            "echo": False,
        }

        try:
            logger.debug(f"Генерація для моделі {model_name}: prompt_len={len(request.prompt)}")
            stream = llm.create_completion(**completion_kwargs)

            for chunk in stream:
                try:
                    # Правильний парсинг відповіді llama-cpp-python
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    
                    choice = choices[0]
                    
                    # Для потокового режиму може бути "delta" або "text"
                    if "delta" in choice:
                        text = choice["delta"].get("content", "")
                    elif "text" in choice:
                        text = choice.get("text", "")
                    else:
                        continue
                    
                    if text:
                        yield text
                        
                except (KeyError, IndexError, TypeError) as e:
                    logger.debug(f"Пропущено чанк: {e}")
                    continue

        except Exception as e:
            logger.error(f"Помилка в generate_stream: {e}", exc_info=True)
            raise

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Метадані моделі"""
        path = self._resolve_path(model_name)
        if not path:
            return None

        try:
            # Базова інформація без завантаження моделі
            info = {
                "name": model_name,
                "path": str(path),
                "type": path.suffix.lower() if path.is_file() else "folder",
            }
            
            if path.is_file():
                info["size"] = path.stat().st_size
                info["size_mb"] = round(path.stat().st_size / (1024 * 1024), 2)
            
            # Спробувати отримати детальні метадані (опціонально)
            try:
                from llama_cpp import Llama
                llm = Llama(str(path), n_ctx=512, verbose=False, n_gpu_layers=0)
                metadata = llm.metadata
                info.update({
                    "architecture": metadata.get("general.architecture", "unknown"),
                    "quantization": metadata.get("general.quantization_version", "unknown"),
                    "context_length": metadata.get("llama.context_length", "unknown"),
                })
                llm.close()
            except Exception as e:
                logger.debug(f"Не вдалося отримати детальні метадані: {e}")
            
            return info
            
        except Exception as e:
            logger.warning(f"Помилка отримання інформації про {model_name}: {e}")
            return {
                "name": model_name,
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """Вивантаження всіх моделей"""
        for name, llm in self.loaded_models.items():
            try:
                llm.close()
                logger.info(f"✓ Вивантажено модель: {name}")
            except Exception as e:
                logger.error(f"✗ Помилка вивантаження {name}: {e}")
        self.loaded_models.clear()