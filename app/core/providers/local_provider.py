# app/core/providers/local_provider.py
import os
from pathlib import Path
from typing import List, AsyncGenerator, Dict, Any, Optional
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.core.providers.base import BaseLLMProvider
from app.config import settings
import logging
import time

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

        # Шукаємо ВСІ .gguf, .bin, .pt файли рекурсивно
        for ext in ("*.gguf", "*.bin", "*.pt"):
            for file in self.models_dir.rglob(ext):
                if file.is_file():
                    # Назва моделі = шлях відносно models_dir без розширення
                    rel_path = file.relative_to(self.models_dir).parent / file.stem
                    model_name = str(rel_path).replace("\\", "/")
                    if model_name.startswith("./"):
                        model_name = model_name[2:]
                    if model_name.endswith("/"):  # якщо папка
                        model_name = model_name[:-1]
                    models.add(model_name)

        # Додаємо HF-папки (навіть без файлів — якщо є config.json)
        for dirpath in self.models_dir.rglob("*"):
            if dirpath.is_dir():
                if (dirpath / "config.json").exists() or any(
                    f.suffix in {".safetensors", ".bin", ".pt"} for f in dirpath.iterdir()
                ):
                    rel = dirpath.relative_to(self.models_dir)
                    models.add(str(rel).replace("\\", "/"))

        result = sorted(models)
        logger.info(f"Знайдено {len(result)} локальних моделей: {result}")
        return result

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
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        model_path = self._resolve_path(model_name)
        if not model_path:
            raise ValueError(f"Модель '{model_name}' не знайдена")

        from llama_cpp import Llama

        n_gpu_layers = self.config.get("gpu_layers", -1)
        if n_gpu_layers == -1:
            n_gpu_layers = 999

        llm = Llama(
            model_path=str(model_path),
            n_ctx=settings.context_size or 8192,
            n_batch=512,
            n_threads=settings.n_threads or max(1, os.cpu_count() - 1),
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            logits_all=False,
            use_mlock=True,
            use_mmap=True,
            # Ключ до стабільності Instruct-модель
            chat_format="llama-3" if "llama-3" in model_name.lower() else None,
        )

        self.loaded_models[model_name] = llm
        logger.info(f"Завантажено модель: {model_name}")
        return llm

    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        available_models = await self.list_models()
        model_name = request.model or (available_models[0] if available_models else None)
        if not model_name:
            raise ValueError("Немає доступних моделей")

        llm = self._get_llama(model_name)

        # КРИТИЧНО: примусово вмикаємо правильний шаблон для всіх Llama-3/3.1/3.2
        force_chat_template = any(x in model_name.lower() for x in ["llama-3", "llama3"])

        try:
            # Формуємо messages у правильному форматі
            messages = []

            # Системне повідомлення (якщо є)
            if request.system_message:
                messages.append({"role": "system", "content": request.system_message.strip()})
            else:
                # Дефолтна система для Llama-3 — без неї модель часто з’їжджає
                messages.append({"role": "system", "content": "Ти — корисний і дружній асистент."})

            # Додаємо контекст, якщо є
            if request.context:
                for msg in request.context:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role in {"user", "assistant", "system"} and content:
                        messages.append({"role": role, "content": content})

            # Останнє повідомлення користувача
            messages.append({"role": "user", "content": request.prompt.strip()})

            # === УНІВЕРСАЛЬНИЙ CHAT COMPLETION ===
            stream = llm.create_chat_completion(
                messages=messages,
                max_tokens=request.max_tokens or 1024,
                temperature=request.temperature or 0.4,
                top_p=request.top_p or 0.95,
                top_k=request.top_k or 40,
                repeat_penalty=1.1,
                stop=request.stop_sequences or None,
                stream=True,
            )

            for chunk in stream:
                content = chunk["choices"][0]["delta"].get("content")
                if content is not None:
                    logger.info(f"✓ Генеровано {len(content)} символів: {content}")
                    yield content

                if chunk["choices"][0].get("finish_reason") is not None:
                    break

        except Exception as e:
            logger.error(f"Помилка генерації ({model_name}): {e}", exc_info=True)
            raise
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Звичайна (не потокова) генерація"""
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
            provider="local_provider",
            generation_time=round(generation_time, 3),
        )
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Метадані моделі"""
        path = self._resolve_path(model_name)
        if not path:
            return None

        try:
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