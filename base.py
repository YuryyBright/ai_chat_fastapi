import os
from pathlib import Path

def create_project_structure(base_dir: str = "llm-service"):
    """
    Створює повну структуру директорій та файлів для проєкту llm-service
    """
    base = Path(base_dir)
    
    # Список всіх директорій
    directories = [
        base / "app",
        base / "app/api/routes",
        base / "app/core/providers",
        base / "app/schemas",
        base / "models/huggingface",
        base / "models/fine-tuned",
        base / "tests",
        base / "data",
    ]
    
    # Створюємо директорії
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Створено директорію: {dir_path}")

    # Список файлів з вмістом (або порожніх)
    files = {
        # app/
        "app/__init__.py": "",
        "app/main.py": "# FastAPI application entry point\nfrom fastapi import FastAPI\n\napp = FastAPI(title=\"LLM Service\")\n",
        "app/config.py": "# Pydantic configuration settings\nfrom pydantic_settings import BaseSettings\n\nclass Settings(BaseSettings):\n    pass\n\nsettings = Settings()\n",
        
        # api/
        "app/api/__init__.py": "",
        
        # routes
        "app/api/routes/__init__.py": "",
        "app/api/routes/generation.py": "# Text generation endpoints\n",
        "app/api/routes/models.py": "# Model management endpoints\n",
        "app/api/routes/training.py": "# Training endpoints\n",
        
        # core/
        "app/core/__init__.py": "",
        "app/core/exceptions.py": "# Custom exceptions\n",
        "app/core/training.py": "# Training manager\n",
        
        # providers
        "app/core/providers/__init__.py": "# Provider manager\n",
        "app/core/providers/base.py": "# Base provider interface\nfrom abc import ABC, abstractmethod\n\nclass BaseProvider(ABC):\n    @abstractmethod\n    async def generate(self, prompt: str, **kwargs):\n        pass\n",
        "app/core/providers/ollama_provider.py": "# Ollama provider implementation\n",
        "app/core/providers/huggingface_provider.py": "# HuggingFace provider implementation\n",
        "app/core/providers/openai_provider.py": "# OpenAI provider implementation\n",
        
        # schemas
        "app/schemas/__init__.py": "",
        "app/schemas/generation.py": "# Generation schemas\nfrom pydantic import BaseModel\n",
        "app/schemas/models.py": "# Model schemas\nfrom pydantic import BaseModel\n",
        "app/schemas/training.py": "# Training schemas\nfrom pydantic import BaseModel\n",
        
        # data & models
        "data/.gitkeep": "",  # щоб папка не ігнорувалася в git
        "models/huggingface/.gitkeep": "",
        "models/fine-tuned/.gitkeep": "",
        
        # tests
        "tests/__init__.py": "",
        "tests/test_generation.py": "",
        "tests/test_models.py": "",
        "tests/test_training.py": "",
        
        # кореневі файли
        "Dockerfile": "# Docker configuration\nFROM python:3.11-slim\n",
        "docker-compose.yml": "# Docker Compose configuration\nservices:\n  llm-service:\n    build: .\n",
        "requirements.txt": "fastapi\nuvicorn[standard]\npydantic-settings\n# додай свої залежності\n",
        ".env.example": "# Environment variables example\nOLLAMA_HOST=http://localhost:11434\n",
        ".dockerignore": "__pycache__\n*.pyc\n.env\n",
        ".gitignore": "__pycache__/\n*.pyc\n.env\nvenv/\n.data/\nmodels/**\n!models/**/.gitkeep\n",
        "README.md": "# LLM Service\n\nСервіс для генерації тексту, файн-тюнінгу та роботи з різними LLM-провайдерами.\n",
    }

    # Створюємо файли
    for file_path, content in files.items():
        full_path = base / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Якщо файл вже існує — пропускаємо (або можна перезаписати)
        if full_path.exists():
            print(f"Файл вже існує, пропускаємо: {full_path}")
            continue
            
        full_path.write_text(content, encoding="utf-8")
        print(f"Створено файл: {full_path}")

    print("\nГотово! Структура проєкту llm-service успішно створена в папці:", base.resolve())


if __name__ == "__main__":
    # Запусти скрипт з кореня, де хочеш створити проєкт
    create_project_structure()