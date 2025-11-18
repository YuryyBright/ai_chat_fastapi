# app/main.py
"""
Головний файл FastAPI додатку для локальної роботи з LLM
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.core.providers import ProviderManager
from app.core.model_downloader import ModelDownloader
from app.api.routes import generation, models

# Налаштування логування
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager для додатку
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"🚀 Запуск {settings.app_name} v{settings.version}")
    logger.info("=" * 60)
    
    # Ініціалізація провайдерів
    logger.info("📦 Ініціалізація локального провайдера...")
    provider_manager = ProviderManager()
    await provider_manager.initialize()
    app.state.provider_manager = provider_manager
    
    # Ініціалізація завантажувача моделей
    logger.info("📥 Ініціалізація завантажувача моделей...")
    model_downloader = ModelDownloader(models_dir=settings.models_dir)
    app.state.model_downloader = model_downloader
    
    # Перевірка доступних моделей
    all_models = await provider_manager.get_all_models()
    total_models = sum(len(models) for models in all_models.values())
    logger.info(f"✓ Знайдено {total_models} локальних моделей")
    
    if total_models == 0:
        logger.warning("⚠️  Немає локальних моделей!")
        logger.info("💡 Завантажте моделі через API /models/download або розмістіть їх в директорії ./models")
    
    logger.info("=" * 60)
    logger.info(f"✓ Сервер готовий: http://{settings.host}:{settings.port}")
    logger.info(f"📚 Документація: http://{settings.host}:{settings.port}/docs")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("🛑 Зупинка сервера...")
    await provider_manager.cleanup()
    logger.info("✓ Ресурси звільнено")


# Створення FastAPI додатку
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Локальний сервіс для роботи з LLM моделями без зовнішніх API",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Підключення роутів
app.include_router(
    generation.router,
    prefix="/generate",
    tags=["Generation"]
)

app.include_router(
    models.router,
    prefix="/models",
    tags=["Models"]
)


@app.get("/")
async def root():
    """
    Кореневий ендпоінт
    """
    return {
        "name": settings.app_name,
        "version": settings.version,
        "status": "online",
        "mode": "local_only",
        "docs": "/docs",
        "endpoints": {
            "models": "/models/list",
            "health": "/models/health",
            "generate": "/generate",
            "stream": "/generate/stream",
            "download": "/models/download",
            "search": "/models/search"
        }
    }


@app.get("/health")
async def health_check():
    """
    Перевірка здоров'я сервісу
    """
    provider_manager: ProviderManager = app.state.provider_manager
    health_status = await provider_manager.health_check()
    
    return {
        "status": "healthy" if all(health_status.values()) else "degraded",
        "providers": health_status,
        "version": settings.version
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development"
    )