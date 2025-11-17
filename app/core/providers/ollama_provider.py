# app/core/providers/ollama_provider.py
"""
Ollama provider implementation
"""

import aiohttp
import time
import json
from typing import AsyncGenerator, List, Dict, Any
from app.core.providers.base import BaseLLMProvider
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.core.exceptions import ProviderError
import logging

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Provider for Ollama local models
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.timeout = config.get("timeout", 300)
        self.default_model = config.get("default_model", "llama2")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self) -> None:
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        logger.info(f"Ollama provider initialized: {self.base_url}")
    
    async def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Ollama availability check failed: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return [model["name"] for model in data.get("models", [])]
                return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    async def generate(
        self,
        request: GenerationRequest
    ) -> GenerationResponse:
        """Generate text using Ollama"""
        start_time = time.time()
        model = request.model or self.default_model
        
        payload = {
            "model": model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
            }
        }
        
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens
        
        if request.top_k:
            payload["options"]["top_k"] = request.top_k
        
        if request.stop_sequences:
            payload["options"]["stop"] = request.stop_sequences
        
        if request.system_message:
            payload["system"] = request.system_message
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ProviderError(f"Ollama error: {error_text}")
                
                data = await response.json()
                generation_time = time.time() - start_time
                
                return GenerationResponse(
                    generated_text=data.get("response", ""),
                    model=model,
                    provider="ollama",
                    tokens_used=data.get("eval_count"),
                    finish_reason=data.get("done_reason"),
                    generation_time=generation_time,
                    metadata={
                        "context": data.get("context"),
                        "total_duration": data.get("total_duration"),
                        "load_duration": data.get("load_duration"),
                        "prompt_eval_count": data.get("prompt_eval_count"),
                    }
                )
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise ProviderError(f"Ollama generation failed: {str(e)}")
    
    async def generate_stream(
        self,
        request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming using Ollama"""
        model = request.model or self.default_model
        
        payload = {
            "model": model,
            "prompt": request.prompt,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
            }
        }
        
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens
        
        if request.system_message:
            payload["system"] = request.system_message
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ProviderError(f"Ollama streaming error: {error_text}")
                
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                yield chunk["response"]
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise ProviderError(f"Ollama streaming failed: {str(e)}")
    
    async def cleanup(self) -> None:
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            logger.info("Ollama provider cleaned up")