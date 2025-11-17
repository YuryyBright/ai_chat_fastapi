# app/schemas/generation.py
"""
Pydantic schemas for text generation requests and responses
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


class GenerationRequest(BaseModel):
    """Schema for text generation request"""
    
    prompt: str = Field(
        ...,
        description="Input prompt for text generation",
        min_length=1,
        max_length=10000
    )
    
    provider: Literal["ollama", "huggingface", "openai"] = Field(
        default="ollama",
        description="LLM provider to use"
    )
    
    model: Optional[str] = Field(
        default=None,
        description="Model name (uses default if not specified)"
    )
    
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=32768,
        description="Maximum tokens to generate"
    )
    
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    
    top_p: Optional[float] = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Top-k sampling parameter"
    )
    
    stream: bool = Field(
        default=False,
        description="Enable streaming response"
    )
    
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="Stop generation at these sequences"
    )
    
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty for token repetition"
    )
    
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty for token repetition"
    )
    
    system_message: Optional[str] = Field(
        default=None,
        description="System message for chat models"
    )
    
    context: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Conversation context for chat models"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain quantum computing in simple terms",
                "provider": "ollama",
                "model": "llama2",
                "max_tokens": 500,
                "temperature": 0.7,
                "stream": False
            }
        }


class GenerationResponse(BaseModel):
    """Schema for text generation response"""
    
    generated_text: str = Field(
        ...,
        description="Generated text"
    )
    
    model: str = Field(
        ...,
        description="Model used for generation"
    )
    
    provider: str = Field(
        ...,
        description="Provider used"
    )
    
    tokens_used: Optional[int] = Field(
        default=None,
        description="Number of tokens used"
    )
    
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason for generation completion"
    )
    
    generation_time: float = Field(
        ...,
        description="Generation time in seconds"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )


class StreamChunk(BaseModel):
    """Schema for streaming response chunk"""
    
    text: str = Field(
        ...,
        description="Text chunk"
    )
    
    done: bool = Field(
        default=False,
        description="Indicates if generation is complete"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Chunk metadata"
    )


# app/schemas/models.py
"""
Pydantic schemas for model management
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ModelInfo(BaseModel):
    """Schema for model information"""
    
    name: str = Field(
        ...,
        description="Model name"
    )
    
    provider: str = Field(
        ...,
        description="Provider name"
    )
    
    type: str = Field(
        ...,
        description="Model type (e.g., 'chat', 'completion')"
    )
    
    size: Optional[str] = Field(
        default=None,
        description="Model size (e.g., '7B', '13B')"
    )
    
    capabilities: List[str] = Field(
        default_factory=list,
        description="Model capabilities"
    )
    
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Model parameters"
    )
    
    status: str = Field(
        default="available",
        description="Model status"
    )
    
    last_updated: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp"
    )


class ModelListResponse(BaseModel):
    """Schema for model list response"""
    
    models: List[ModelInfo] = Field(
        default_factory=list,
        description="List of available models"
    )
    
    total: int = Field(
        ...,
        description="Total number of models"
    )


