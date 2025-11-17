# tests/conftest.py
"""
Pytest configuration and fixtures
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_generation_request():
    """Sample generation request data"""
    return {
        "prompt": "Test prompt",
        "provider": "ollama",
        "model": "llama2",
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }


@pytest.fixture
def sample_training_request():
    """Sample training request data"""
    return {
        "base_model": "gpt2",
        "provider": "huggingface",
        "dataset": {
            "texts": [
                "Sample text 1",
                "Sample text 2",
                "Sample text 3"
            ]
        },
        "output_name": "test-model",
        "epochs": 1,
        "batch_size": 2
    }
