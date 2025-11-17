# tests/test_generation.py
"""
Tests for generation endpoints
"""

import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert data["status"] == "operational"


def test_health_check(client: TestClient):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "providers" in data


def test_generate_validation_error(client: TestClient):
    """Test generation with invalid request"""
    response = client.post(
        "/api/v1/generation/generate",
        json={"prompt": ""}  # Empty prompt should fail
    )
    assert response.status_code == 422


def test_generate_with_stream_flag(client: TestClient):
    """Test that stream flag redirects to proper endpoint"""
    response = client.post(
        "/api/v1/generation/generate",
        json={
            "prompt": "Test",
            "provider": "ollama",
            "stream": True
        }
    )
    assert response.status_code == 400
    assert "stream" in response.json()["detail"].lower()


def test_invalid_provider(client: TestClient):
    """Test generation with invalid provider"""
    response = client.post(
        "/api/v1/generation/generate",
        json={
            "prompt": "Test prompt",
            "provider": "invalid_provider"
        }
    )
    assert response.status_code in [422, 503]


@pytest.mark.asyncio
async def test_generation_response_structure(client: TestClient, sample_generation_request):
    """Test that successful generation has correct structure"""
    # Note: This test requires Ollama to be running
    # In real scenarios, use mocking for unit tests
    response = client.post(
        "/api/v1/generation/generate",
        json=sample_generation_request
    )
    
    # If Ollama is not available, test should skip
    if response.status_code == 503:
        pytest.skip("Ollama service not available")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "generated_text" in data
    assert "model" in data
    assert "provider" in data
    assert "generation_time" in data
    assert "timestamp" in data
