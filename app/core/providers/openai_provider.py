# app/core/providers/openai_provider.py
"""
OpenAI provider implementation
"""

import time
import openai
from typing import AsyncGenerator, List, Dict, Any
from app.core.providers.base import BaseLLMProvider
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.core.exceptions import ProviderError
import logging

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    Provider for OpenAI GPT models
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.organization = config.get("organization")
        self.default_model = config.get("default_model", "gpt-3.5-turbo")
        self.client = None
    
    async def initialize(self) -> None:
        """Initialize OpenAI client"""
        if not self.api_key:
            logger.warning("OpenAI API key not provided")
            return
        
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            organization=self.organization
        )
        logger.info("OpenAI provider initialized")
    
    async def is_available(self) -> bool:
        """Check if OpenAI is available"""
        if not self.client or not self.api_key:
            return False
        
        try:
            await self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI availability check failed: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List available OpenAI models"""
        if not self.client:
            return []
        
        try:
            models = await self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {e}")
            return []
    
    async def generate(
        self,
        request: GenerationRequest
    ) -> GenerationResponse:
        """Generate text using OpenAI"""
        if not self.client:
            raise ProviderError("OpenAI client not initialized")
        
        start_time = time.time()
        model = request.model or self.default_model
        
        # Prepare messages
        messages = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        
        if request.context:
            messages.extend(request.context)
        
        messages.append({"role": "user", "content": request.prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                stop=request.stop_sequences
            )
            
            generation_time = time.time() - start_time
            choice = response.choices[0]
            
            return GenerationResponse(
                generated_text=choice.message.content,
                model=model,
                provider="openai",
                tokens_used=response.usage.total_tokens,
                finish_reason=choice.finish_reason,
                generation_time=generation_time,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
        
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise ProviderError(f"OpenAI generation failed: {str(e)}")
    
    async def generate_stream(
        self,
        request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming using OpenAI"""
        if not self.client:
            raise ProviderError("OpenAI client not initialized")
        
        model = request.model or self.default_model
        
        # Prepare messages
        messages = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        
        if request.context:
            messages.extend(request.context)
        
        messages.append({"role": "user", "content": request.prompt})
        
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                stop=request.stop_sequences,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise ProviderError(f"OpenAI streaming failed: {str(e)}")