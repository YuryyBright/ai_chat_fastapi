# app/core/providers/base.py
"""
Base provider interface for LLM implementations
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Dict, Any, List
from app.schemas.generation import GenerationRequest, GenerationResponse


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    All providers must implement these methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize provider with configuration
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize provider resources"""
        pass
    
    @abstractmethod
    async def generate(
        self,
        request: GenerationRequest
    ) -> GenerationResponse:
        """
        Generate text synchronously
        
        Args:
            request: Generation request parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate text with streaming
        
        Args:
            request: Generation request parameters
            
        Yields:
            Text chunks as they're generated
        """
        pass
    
    @abstractmethod
    async def list_models(self) -> List[str]:
        """
        List available models for this provider
        
        Returns:
            List of model names
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if provider is available and operational
        
        Returns:
            True if provider is ready
        """
        pass
    
    async def cleanup(self) -> None:
        """Cleanup provider resources"""
        pass
