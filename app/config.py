# app/config.py
"""
Configuration Management using Pydantic Settings

Single source of truth for all application configuration.
Supports environment variables and .env files.

ONLY LOCAL MODELS - No external API dependencies
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import List, Literal, Optional
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
    
    # ============================================
    # Application Settings
    # ============================================
    app_name: str = Field(
        default="Local LLM Service",
        description="Application name"
    )
    version: str = Field(
        default="2.0.0",
        description="Application version"
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="production",
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
    
    # ============================================
    # CORS Settings
    # ============================================
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    # ============================================
    # Provider Settings (Local Only)
    # ============================================
    enabled_providers: List[str] = Field(
        default=["local_provider"],
        description="Enabled providers (only local_provider supported)"
    )
    
    # ============================================
    # Storage Settings
    # ============================================
    data_dir: str = Field(
        default="./data",
        description="Data storage directory"
    )
    models_dir: str = Field(
        default="./models",
        description="Models storage directory"
    )
    cache_dir: str = Field(
        default="./models/cache",
        description="Cache directory for downloads"
    )
    
    # ============================================
    # Local Unified Provider Configuration
    # ============================================
    context_size: int = Field(
        default=8192,
        ge=512,
        le=32768,
        description="Context window size in tokens"
    )
    batch_size: int = Field(
        default=512,
        ge=32,
        le=2048,
        description="Batch size for processing"
    )
    n_threads: Optional[int] = Field(
        default=None,
        description="Number of CPU threads (None = auto)"
    )
    use_gpu: bool = Field(
        default=True,
        description="Enable GPU acceleration (CUDA)"
    )
    gpu_layers: int = Field(
        default=-1,
        description="Number of layers to offload to GPU (-1 = all, 0 = CPU only)"
    )
    
    # ============================================
    # Generation Settings (Defaults)
    # ============================================
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
    default_top_k: int = Field(
        default=40,
        ge=1,
        le=100,
        description="Default top_k for generation"
    )
    
    # ============================================
    # Model Download Settings
    # ============================================
    huggingface_token: Optional[str] = Field(
        default=None,
        description="HuggingFace token for private models"
    )
    download_timeout: int = Field(
        default=3600,
        ge=60,
        description="Download timeout in seconds"
    )
    max_download_size: int = Field(
        default=50,
        ge=1,
        description="Maximum download size in GB"
    )
    
    # ============================================
    # Security Settings
    # ============================================
    api_key_enabled: bool = Field(
        default=False,
        description="Enable API key authentication"
    )
    api_keys: List[str] = Field(
        default=[],
        description="Valid API keys"
    )
    
    # ============================================
    # Rate Limiting
    # ============================================
    rate_limit_enabled: bool = Field(
        default=False,
        description="Enable rate limiting"
    )
    rate_limit_requests: int = Field(
        default=100,
        description="Rate limit: requests per minute"
    )
    
    # ============================================
    # Training Settings (для майбутніх розширень)
    # ============================================
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
    
    # ============================================
    # Validators
    # ============================================
    @field_validator("enabled_providers")
    @classmethod
    def validate_providers(cls, v):
        """Validate that only local_provider provider is enabled"""
        supported = ["local_provider"]
        for provider in v:
            if provider not in supported:
                raise ValueError(
                    f"Unsupported provider: {provider}. "
                    f"Only 'local_provider' is supported in this version."
                )
        return v
    
    @field_validator("gpu_layers")
    @classmethod
    def validate_gpu_layers(cls, v, info):
        """Validate GPU layers setting"""
        if v < -1:
            raise ValueError("gpu_layers must be -1 (all), 0 (CPU), or positive number")
        return v
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training_output_dir).mkdir(parents=True, exist_ok=True)
    
    def get_llama_cpp_config(self) -> dict:
        """Get configuration for llama-cpp-python"""
        return {
            "models_dir": self.models_dir,
            "context_size": self.context_size,
            "batch_size": self.batch_size,
            "threads": self.n_threads,
            "use_gpu": self.use_gpu,
            "gpu_layers": self.gpu_layers,
        }
    
    def log_configuration(self):
        """Log current configuration (for debugging)"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("=" * 60)
        logger.info("Configuration:")
        logger.info(f"  App: {self.app_name} v{self.version}")
        logger.info(f"  Environment: {self.environment}")
        logger.info(f"  Models directory: {self.models_dir}")
        logger.info(f"  Context size: {self.context_size}")
        logger.info(f"  GPU enabled: {self.use_gpu}")
        if self.use_gpu:
            logger.info(f"  GPU layers: {self.gpu_layers}")
        logger.info(f"  CPU threads: {self.n_threads or 'auto'}")
        logger.info("=" * 60)


# Global settings instance
settings = Settings()