# app/core/providers/huggingface_provider.py
"""
HuggingFace provider implementation
"""

import time
import torch
from typing import AsyncGenerator, List, Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline
)
from threading import Thread
from app.core.providers.base import BaseLLMProvider
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.core.exceptions import ProviderError
import logging

logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseLLMProvider):
    """
    Provider for HuggingFace models (local and API)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cache_dir = config.get("cache_dir", "./models/huggingface")
        self.default_model = config.get("default_model", "gpt2")
        self.api_key = config.get("api_key")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_tokenizers: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize HuggingFace provider"""
        logger.info(f"HuggingFace provider initialized on device: {self.device}")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    async def is_available(self) -> bool:
        """Check if HuggingFace is available"""
        return True  # Always available if transformers is installed
    
    async def list_models(self) -> List[str]:
        """List loaded models"""
        return list(self.loaded_models.keys())
    
    def _load_model(self, model_name: str) -> tuple:
        """
        Load model and tokenizer
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name], self.loaded_tokenizers[model_name]
        
        try:
            logger.info(f"Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                token=self.api_key
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                token=self.api_key,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            self.loaded_models[model_name] = model
            self.loaded_tokenizers[model_name] = tokenizer
            
            logger.info(f"Model loaded successfully: {model_name}")
            return model, tokenizer
        
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ProviderError(f"Failed to load model: {str(e)}")
    
    async def generate(
        self,
        request: GenerationRequest
    ) -> GenerationResponse:
        """Generate text using HuggingFace model"""
        start_time = time.time()
        model_name = request.model or self.default_model
        
        try:
            model, tokenizer = self._load_model(model_name)
            
            # Prepare generation config
            generation_config = {
                "max_new_tokens": request.max_tokens or 512,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "do_sample": True,
            }
            
            if request.top_k:
                generation_config["top_k"] = request.top_k
            
            # Tokenize input
            inputs = tokenizer(
                request.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_config,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            generation_time = time.time() - start_time
            
            return GenerationResponse(
                generated_text=generated_text,
                model=model_name,
                provider="huggingface",
                tokens_used=len(outputs[0]),
                generation_time=generation_time,
                metadata={
                    "device": self.device,
                    "input_tokens": inputs['input_ids'].shape[1]
                }
            )
        
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise ProviderError(f"HuggingFace generation failed: {str(e)}")
    
    async def generate_stream(
        self,
        request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming using HuggingFace"""
        model_name = request.model or self.default_model
        
        try:
            model, tokenizer = self._load_model(model_name)
            
            # Create streamer
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Prepare generation config
            generation_config = {
                "max_new_tokens": request.max_tokens or 512,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "do_sample": True,
                "streamer": streamer,
            }
            
            if request.top_k:
                generation_config["top_k"] = request.top_k
            
            # Tokenize input
            inputs = tokenizer(
                request.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Start generation in thread
            generation_kwargs = {**inputs, **generation_config}
            generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
            
            thread = Thread(
                target=model.generate,
                kwargs=generation_kwargs
            )
            thread.start()
            
            # Yield tokens as they're generated
            for text in streamer:
                yield text
            
            thread.join()
        
        except Exception as e:
            logger.error(f"HuggingFace streaming error: {e}")
            raise ProviderError(f"HuggingFace streaming failed: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup loaded models"""
        self.loaded_models.clear()
        self.loaded_tokenizers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("HuggingFace provider cleaned up")