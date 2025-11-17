# app/config.py
"""
Configuration Management using Pydantic Settings

Single source of truth for all application configuration.
Supports environment variables and .env files.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from typing import List, Literal
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be overridden using environment variables.
    Example: APP_NAME -> app_name
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application Settings
    app_name: str = Field(
        default="LLM Service",
        description="Application name"
    )
    version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime environment"
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind the service"
    )
    port: int = Field(
        default=8000,
        description="Port to bind the service"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    # CORS Settings
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    # Provider Settings
    enabled_providers: List[str] = Field(
        default=["ollama", "huggingface", "openai"],
        description="List of enabled LLM providers"
    )
    
    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama service base URL"
    )
    ollama_timeout: int = Field(
        default=300,
        description="Ollama request timeout in seconds"
    )
    ollama_default_model: str = Field(
        default="llama2",
        description="Default Ollama model"
    )
    
    # HuggingFace Configuration
    huggingface_api_key: str = Field(
        default="",
        description="HuggingFace API key"
    )
    huggingface_cache_dir: str = Field(
        default="./models/huggingface",
        description="HuggingFace models cache directory"
    )
    huggingface_default_model: str = Field(
        default="gpt2",
        description="Default HuggingFace model"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key"
    )
    openai_organization: str = Field(
        default="",
        description="OpenAI organization ID"
    )
    openai_default_model: str = Field(
        default="gpt-3.5-turbo",
        description="Default OpenAI model"
    )
    
    # Local Universal Provider (tabbyAPI, llama.cpp, vLLM, etc.)
    local_base_url: str = Field(
        default="http://127.0.0.1:8080/v1",
        description="Base URL for any OpenAI-compatible local server"
    )
    local_api_key: str = Field(default="not-needed")
    local_timeout: int = Field(default=600)
    local_default_model: str = Field(default="local-model")
    # Generation Settings
    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=32768,
        description="Maximum tokens for generation"
    )
    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for generation"
    )
    default_top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Default top_p for generation"
    )
    
    # Training Settings
    training_output_dir: str = Field(
        default="./models/fine-tuned",
        description="Directory for fine-tuned models"
    )
    max_training_epochs: int = Field(
        default=10,
        ge=1,
        description="Maximum training epochs"
    )
    training_batch_size: int = Field(
        default=8,
        ge=1,
        description="Training batch size"
    )
    learning_rate: float = Field(
        default=2e-5,
        gt=0.0,
        description="Learning rate for training"
    )
    
    # Storage Settings
    data_dir: str = Field(
        default="./data",
        description="Data storage directory"
    )
    models_dir: str = Field(
        default="./models",
        description="Models storage directory"
    )
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    rate_limit_requests: int = Field(
        default=100,
        description="Rate limit: requests per minute"
    )
    
    # Security
    api_key_enabled: bool = Field(
        default=False,
        description="Enable API key authentication"
    )
    api_keys: List[str] = Field(
        default=[],
        description="Valid API keys"
    )
    
    @validator("enabled_providers")
    def validate_providers(cls, v):
        """Validate that enabled providers are supported"""
        supported = ["ollama", "huggingface", "openai"]
        for provider in v:
            if provider not in supported:
                raise ValueError(f"Unsupported provider: {provider}")
        return v
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.huggingface_cache_dir).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()