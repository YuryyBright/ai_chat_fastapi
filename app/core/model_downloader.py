# app/core/model_downloader.py
"""
Система завантаження моделей з HuggingFace
"""

import os
import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ModelDownloader:
    """
    Завантажувач моделей з HuggingFace
    """
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    async def download_model(
        self,
        repo_id: str,
        model_type: str = "gguf",
        filename: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Завантаження моделі з HuggingFace
        
        Args:
            repo_id: ID репозиторію (наприклад, "TheBloke/Llama-2-7B-GGUF")
            model_type: Тип моделі ("gguf", "huggingface", "gptq", "awq")
            filename: Конкретний файл для завантаження (для GGUF)
        
        Yields:
            Прогрес завантаження
        """
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            
            if model_type == "gguf":
                # Завантаження GGUF файлу
                yield {
                    "status": "starting",
                    "message": f"Початок завантаження GGUF моделі {repo_id}",
                    "progress": 0
                }
                
                if not filename:
                    raise ValueError("Для GGUF моделей потрібно вказати filename")
                
                # Завантаження в окремому потоці
                loop = asyncio.get_event_loop()
                
                def download_sync():
                    return hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=str(self.models_dir / "cache"),
                        local_dir=str(self.models_dir),
                        local_dir_use_symlinks=False
                    )
                
                # Симуляція прогресу (huggingface_hub не дає точний прогрес)
                download_task = loop.run_in_executor(self.executor, download_sync)
                
                progress = 0
                while not download_task.done():
                    progress = min(progress + 5, 95)
                    yield {
                        "status": "downloading",
                        "message": f"Завантаження {filename}...",
                        "progress": progress
                    }
                    await asyncio.sleep(2)
                
                model_path = await download_task
                
                yield {
                    "status": "complete",
                    "message": f"Модель успішно завантажена: {model_path}",
                    "progress": 100,
                    "model_path": str(model_path)
                }
                
            elif model_type in ["huggingface", "gptq", "awq", "exl2"]:
                # Завантаження повної папки моделі
                yield {
                    "status": "starting",
                    "message": f"Початок завантаження моделі {repo_id}",
                    "progress": 0
                }
                
                model_name = repo_id.split("/")[-1]
                local_dir = self.models_dir / model_name
                
                loop = asyncio.get_event_loop()
                
                def download_snapshot_sync():
                    return snapshot_download(
                        repo_id=repo_id,
                        cache_dir=str(self.models_dir / "cache"),
                        local_dir=str(local_dir),
                        local_dir_use_symlinks=False,
                        resume_download=True
                    )
                
                download_task = loop.run_in_executor(self.executor, download_snapshot_sync)
                
                progress = 0
                while not download_task.done():
                    progress = min(progress + 3, 95)
                    yield {
                        "status": "downloading",
                        "message": f"Завантаження {model_name}...",
                        "progress": progress
                    }
                    await asyncio.sleep(3)
                
                model_path = await download_task
                
                yield {
                    "status": "complete",
                    "message": f"Модель успішно завантажена: {model_path}",
                    "progress": 100,
                    "model_path": str(model_path)
                }
            
            else:
                yield {
                    "status": "error",
                    "message": f"Невідомий тип моделі: {model_type}",
                    "progress": 0
                }
                
        except ImportError:
            yield {
                "status": "error",
                "message": "huggingface_hub не встановлений. Встановіть: pip install huggingface_hub",
                "progress": 0
            }
        except Exception as e:
            logger.error(f"Помилка завантаження моделі: {e}")
            yield {
                "status": "error",
                "message": f"Помилка: {str(e)}",
                "progress": 0
            }
    
    async def list_available_models(self, query: str = "") -> list:
        """
        Пошук доступних моделей на HuggingFace
        """
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            
            # Пошук GGUF моделей
            models = api.list_models(
                search=query,
                filter="gguf",
                sort="downloads",
                limit=20
            )
            
            return [
                {
                    "id": model.id,
                    "author": model.author,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "tags": model.tags
                }
                for model in models
            ]
            
        except ImportError:
            logger.error("huggingface_hub не встановлений")
            return []
        except Exception as e:
            logger.error(f"Помилка пошуку моделей: {e}")
            return []
    
    def get_model_files(self, repo_id: str) -> list:
        """
        Отримати список файлів в репозиторії
        """
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            files = api.list_repo_files(repo_id=repo_id)
            
            # Фільтруємо GGUF файли
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            return gguf_files
            
        except ImportError:
            logger.error("huggingface_hub не встановлений")
            return []
        except Exception as e:
            logger.error(f"Помилка отримання файлів: {e}")
            return []
    
    async def delete_model(self, model_name: str) -> bool:
        """
        Видалення локальної моделі
        """
        try:
            model_path = self.models_dir / model_name
            
            if model_path.exists():
                if model_path.is_file():
                    model_path.unlink()
                else:
                    import shutil
                    shutil.rmtree(model_path)
                
                logger.info(f"Модель {model_name} видалена")
                return True
            else:
                logger.warning(f"Модель {model_name} не знайдена")
                return False
                
        except Exception as e:
            logger.error(f"Помилка видалення моделі: {e}")
            return False
    
    def get_model_size(self, model_name: str) -> int:
        """
        Отримати розмір моделі в байтах
        """
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            return 0
        
        if model_path.is_file():
            return model_path.stat().st_size
        
        # Для папки - сумуємо всі файли
        total_size = 0
        for file in model_path.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size
        
        return total_size
    
    def format_size(self, size_bytes: int) -> str:
        """
        Форматування розміру в читабельний вигляд
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"