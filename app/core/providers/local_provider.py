# app/core/providers/local_provider.py
import os
from pathlib import Path
from typing import List, AsyncGenerator, Dict, Any, Optional
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.core.providers.base import BaseLLMProvider
from app.config import settings
import logging

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
                # HuggingFace, GPTQ, AWQ, EXL2 — шукаємо ключові файли
                if (path / "config.json").exists() or any(f.suffix == ".safetensors" for f in path.glob("*.safetensors")):
                    models.add(path.name)

        models_list = sorted(models)
        logger.info(f"Знайдено локальних моделей: {len(models_list)} → {models_list or 'немає'}")
        return models_list

    def _resolve_path(self, model_name: str) -> Optional[Path]:
        candidates = [
            self.models_dir / f"{model_name}.gguf",
            self.models_dir / f"{model_name}.bin",
            self.models_dir / model_name,  # папка
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _get_llama(self, model_name: str):
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

            llm = Llama(
                model_path=str(model_path),
                n_ctx=settings.context_size or 8192,
                n_batch=512,
                n_threads=settings.n_threads or max(1, os.cpu_count() - 1),
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                logits_all=False,           # ВИМКНЕНО — критично для швидкості!
                use_mlock=True,
                use_mmap=True,
                cache_prompt=True,          # прискорює повторні запити

                # Розумні дефолти (не залежать від запиту)
                repeat_penalty=1.1,         # найкраще працює
                top_k=40,
                top_p=0.95,
                temperature=0.7,
                last_n_tokens=64,
            )
            self.loaded_models[model_name] = llm
            logger.info(f"Завантажено модель: {model_name} ← {model_path.name}")
            return llm
        except Exception as e:
            logger.error(f"Помилка завантаження моделі {model_name}: {e}")
            raise

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        result = ""
        async for token in self.generate_stream(request):
            result += token
        return GenerationResponse(generated_text=result.strip())

    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        available_models = await self.list_models()
        model_name = request.model or (available_models[0] if available_models else None)
        
        if not model_name:
            raise ValueError("Немає доступних моделей")

        llm = self._get_llama(model_name)

        # Використовуємо лише ті параметри, які є в твоїй схемі
        completion_kwargs = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature or 0.7,
            "top_p": request.top_p or 0.9,
            "top_k": request.top_k or 40,
            "stop": request.stop_sequences,
            "stream": True,
            # "cache_prompt": True,        # дуже важливо для швидкості
            "echo": False,
        }

        try:
            stream = llm.create_completion(**completion_kwargs)

            for chunk in stream:
                try:
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content") or delta.get("text") or ""
                    if text:
                        yield text
                except (KeyError, IndexError, TypeError):
                    continue  # пропускаємо биті чанки

        except Exception as e:
            logger.error(f"Помилка в generate_stream: {e}", exc_info=True)
            raise
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Повертає метадані моделі (використовується в /info та /list)
        """
        path = self._resolve_path(model_name)
        if not path:
            return None

        try:
            from llama_cpp import Llama
            # Тимчасово створюємо Llama з verbose=False і n_ctx=1 — тільки для метаданих
            llm = Llama(str(path), n_ctx=512, verbose=False, n_gpu_layers=0)
            info = llm.metadata
            llm.close()
            
            return {
                "name": model_name,
                "path": str(path),
                "type": path.suffix.lower() if path.is_file() else "folder",
                "architecture": info.get("architectures", ["unknown"])[0] if info.get("architectures") else "unknown",
                "params": info.get("llama.context_length", "unknown"),
                "quantization": info.get("quantization", "unknown"),
                "size": str(path.stat().st_size) if path.is_file() else "directory",
            }
        except Exception as e:
            logger.warning(f"Не вдалося отримати метадані для {model_name}: {e}")
            return {
                "name": model_name,
                "path": str(path),
                "type": "unknown",
                "error": str(e)
            }

    async def cleanup(self) -> None:
        for name, llm in self.loaded_models.items():
            try:
                llm.close()
                logger.info(f"Вивантажено модель: {name}")
            except:
                pass
        self.loaded_models.clear()