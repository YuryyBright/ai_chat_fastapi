# app/core/providers/__init__.py
"""
Менеджер провайдерів - тільки локальні моделі
"""

from typing import Dict, Optional
import logging

from app.core.providers.base import BaseLLMProvider
from app.core.providers.local_provider import LocalProvider
from app.core.exceptions import ProviderError
from app.config import settings

logger = logging.getLogger(__name__)


class ProviderManager:
    """
    Менеджер для локального провайдера
    Управління моделями без зовнішніх API
    """
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """
        Ініціалізація локального провайдера
        """
        logger.info("Ініціалізація локального провайдера...")
        
        # Створюємо єдиний локальний провайдер з правильною назвою
        self.providers["local_provider"] = LocalProvider({
            "models_dir": settings.models_dir or "./models",
            "context_size": settings.context_size or 8192,
            "batch_size": settings.batch_size or 512,
            "threads": settings.n_threads or None,  # None = auto
            "use_gpu": getattr(settings, 'use_gpu', True),
            "gpu_layers": getattr(settings, 'gpu_layers', -1),  # -1 = всі шари
        })
        
        logger.info("✓ Локальний провайдер створено")
    
    async def initialize(self) -> None:
        """
        Асинхронна ініціалізація всіх провайдерів
        """
        for name, provider in self.providers.items():
            try:
                await provider.initialize()
                logger.info(f"✓ Провайдер '{name}' успішно ініціалізовано")
            except Exception as e:
                logger.error(f"✗ Помилка ініціалізації провайдера '{name}': {e}")
                raise
    
    async def cleanup(self) -> None:
        """
        Очищення всіх провайдерів
        """
        for name, provider in self.providers.items():
            try:
                await provider.cleanup()
                logger.info(f"✓ Провайдер '{name}' очищено")
            except Exception as e:
                logger.error(f"✗ Помилка очищення провайдера '{name}': {e}")
    
    def get_provider(self, provider_name: str = "local_provider") -> BaseLLMProvider:
        """
        Отримати провайдер за іменем
        
        Args:
            provider_name: Ім'я провайдера (за замовчуванням "local_provider")
            
        Returns:
            Інстанс провайдера
            
        Raises:
            ProviderError: Якщо провайдер не знайдено
        """
        if provider_name not in self.providers:
            raise ProviderError(
                f"Провайдер '{provider_name}' не знайдено. "
                f"Доступні провайдери: {list(self.providers.keys())}"
            )
        return self.providers[provider_name]
    
    async def get_all_models(self) -> Dict[str, list]:
        """
        Отримати всі доступні моделі
        
        Returns:
            Словник: {provider_name: [model_names]}
        """
        all_models = {}
        for name, provider in self.providers.items():
            try:
                models = await provider.list_models()
                all_models[name] = models
            except Exception as e:
                logger.error(f"Помилка отримання моделей для '{name}': {e}")
                all_models[name] = []
        return all_models
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Перевірка стану всіх провайдерів
        
        Returns:
            Словник: {provider_name: is_available}
        """
        health_status = {}
        for name, provider in self.providers.items():
            try:
                health_status[name] = await provider.is_available()
            except Exception as e:
                logger.error(f"Помилка перевірки здоров'я '{name}': {e}")
                health_status[name] = False
        return health_status


# Зручний експорт
__all__ = [
    "ProviderManager",
    "BaseLLMProvider",
    "LocalProvider",
]