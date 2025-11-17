# app/core/providers/__init__.py
"""
Provider manager for handling multiple LLM providers
"""

from typing import Dict, Optional
from app.core.providers.base import BaseLLMProvider
from app.core.providers.ollama_provider import OllamaProvider
from app.core.providers.huggingface_provider import HuggingFaceProvider
from app.core.providers.openai_provider import OpenAIProvider
from app.core.providers.local_provider import LocalProvider
from app.config import settings
from app.core.exceptions import ProviderError
import logging

logger = logging.getLogger(__name__)


class ProviderManager:
    """
    Manages multiple LLM providers and routes requests to appropriate provider
    """
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize all enabled providers based on configuration"""
        
        if "ollama" in settings.enabled_providers:
            self.providers["ollama"] = OllamaProvider({
                "base_url": settings.ollama_base_url,
                "timeout": settings.ollama_timeout,
                "default_model": settings.ollama_default_model
            })
        
        if "huggingface" in settings.enabled_providers:
            self.providers["huggingface"] = HuggingFaceProvider({
                "cache_dir": settings.huggingface_cache_dir,
                "default_model": settings.huggingface_default_model,
                "api_key": settings.huggingface_api_key
            })
        
        if "openai" in settings.enabled_providers:
            self.providers["openai"] = OpenAIProvider({
                "api_key": settings.openai_api_key,
                "organization": settings.openai_organization,
                "default_model": settings.openai_default_model
            })
        if "local_provider" in settings.enabled_providers:
            self.providers['local_provider'] = LocalProvider(
                {
                    'local_timeout': settings.local_timeout,
                    'local_default_model': settings.local_default_model
                }
            )
    
    async def initialize(self) -> None:
        """Initialize all providers"""
        for name, provider in self.providers.items():
            try:
                await provider.initialize()
                logger.info(f"Provider '{name}' initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize provider '{name}': {e}")
    
    async def cleanup(self) -> None:
        """Cleanup all providers"""
        for name, provider in self.providers.items():
            try:
                await provider.cleanup()
                logger.info(f"Provider '{name}' cleaned up")
            except Exception as e:
                logger.error(f"Failed to cleanup provider '{name}': {e}")
    
    def get_provider(self, provider_name: str) -> BaseLLMProvider:
        """
        Get provider by name
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider instance
            
        Raises:
            ProviderError: If provider not found or not enabled
        """
        if provider_name not in self.providers:
            raise ProviderError(
                f"Provider '{provider_name}' not found or not enabled. "
                f"Available providers: {list(self.providers.keys())}"
            )
        return self.providers[provider_name]
    
    async def get_all_models(self) -> Dict[str, list]:
        """
        Get all available models from all providers
        
        Returns:
            Dictionary mapping provider names to model lists
        """
        all_models = {}
        for name, provider in self.providers.items():
            try:
                models = await provider.list_models()
                all_models[name] = models
            except Exception as e:
                logger.error(f"Failed to list models for provider '{name}': {e}")
                all_models[name] = []
        return all_models
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health status of all providers
        
        Returns:
            Dictionary mapping provider names to availability status
        """
        health_status = {}
        for name, provider in self.providers.items():
            try:
                health_status[name] = await provider.is_available()
            except Exception as e:
                logger.error(f"Health check failed for provider '{name}': {e}")
                health_status[name] = False
        return health_status