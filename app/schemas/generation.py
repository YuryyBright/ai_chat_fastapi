# app/schemas/generation.py
"""
Pydantic schemas для генерації тексту
Виправлено provider на local_provider
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


class GenerationRequest(BaseModel):
    """Schema для запиту генерації тексту"""
    
    prompt: str = Field(
        ...,
        description="Вхідний промпт для генерації",
        min_length=1,
        max_length=100000
    )
    
    provider: Literal["local_provider"] = Field(
        default="local_provider",
        description="Провайдер LLM (тільки local_provider)"
    )
    
    model: Optional[str] = Field(
        default=None,
        description="Назва моделі (використовується перша доступна, якщо не вказано)"
    )
    
    max_tokens: Optional[int] = Field(
        default=1024,
        ge=1,
        le=32768,
        description="Максимальна кількість токенів для генерації"
    )
    
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Температура семплювання"
    )
    
    top_p: Optional[float] = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling параметр"
    )
    
    top_k: Optional[int] = Field(
        default=40,
        ge=1,
        le=100,
        description="Top-k sampling параметр"
    )
    
    stream: bool = Field(
        default=False,
        description="Увімкнути потокову відповідь"
    )
    
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="Зупинити генерацію на цих послідовностях"
    )
    
    presence_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Штраф за присутність для повторення токенів"
    )
    
    frequency_penalty: Optional[float] = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Штраф за частоту для повторення токенів"
    )
    
    system_message: Optional[str] = Field(
        default=None,
        description="Системне повідомлення для чат-моделей"
    )
    
    context: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Контекст розмови для чат-моделей"
    )
    repeat_penalty: Optional[float] = Field(
        default=1.1,
        ge=0.0,
        le=2.0,
        description="Класичний repeat_penalty (llama.cpp стиль). Використовується, якщо > 0"
    )
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Поясни квантові обчислення простими словами",
                "provider": "local_provider",
                "model": "llama2",
                "max_tokens": 500,
                "temperature": 0.7,
                "stream": False
            }
        }


class GenerationResponse(BaseModel):
    """Schema для відповіді генерації"""
    
    generated_text: str = Field(
        ...,
        description="Згенерований текст"
    )
    
    model: str = Field(
        ...,
        description="Модель, використана для генерації"
    )
    
    provider: str = Field(
        ...,
        description="Використаний провайдер"
    )
    
    tokens_used: Optional[int] = Field(
        default=None,
        description="Кількість використаних токенів"
    )
    
    finish_reason: Optional[str] = Field(
        default=None,
        description="Причина завершення генерації"
    )
    
    generation_time: float = Field(
        ...,
        description="Час генерації в секундах"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Час відповіді"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Додаткові метадані"
    )


class StreamChunk(BaseModel):
    """Schema для чанку потокової відповіді"""
    
    text: str = Field(
        default="",
        description="Текстовий чанк"
    )
    
    done: bool = Field(
        default=False,
        description="Позначає, чи завершена генерація"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Повідомлення про помилку (якщо є)"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Метадані чанку"
    )