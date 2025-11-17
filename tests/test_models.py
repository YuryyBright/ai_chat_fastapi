# tests/test_models.py
"""
Tests for model management endpoints
"""

import pytest
from fastapi.testclient import TestClient


def test_list_all_models(client: TestClient):
    """Test listing all models"""
    response = client.get("/api/v1/models/list")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "total" in data
    assert isinstance(data["models"], list)
    assert isinstance(data["total"], int)


def test_list_provider_models_invalid(client: TestClient):
    """Test listing models for invalid provider"""
    response = client.get("/api/v1/models/list/invalid_provider")
    assert response.status_code == 503


def test_providers_health(client: TestClient):
    """Test providers health check"""
    response = client.get("/api/v1/models/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "providers" in data
    assert isinstance(data["providers"], dict)