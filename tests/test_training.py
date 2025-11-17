# tests/test_training.py
"""
Tests for training endpoints
"""

import pytest
from fastapi.testclient import TestClient


def test_start_training_validation(client: TestClient):
    """Test training with invalid request"""
    response = client.post(
        "/api/v1/training/start",
        json={
            "base_model": "gpt2",
            "provider": "huggingface",
            # Missing required fields
        }
    )
    assert response.status_code == 422


def test_start_training_response_structure(client: TestClient, sample_training_request):
    """Test training start response structure"""
    response = client.post(
        "/api/v1/training/start",
        json=sample_training_request
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "job_id" in data
    assert "status" in data
    assert "progress" in data
    assert data["status"] == "pending"
    assert data["progress"] == 0.0


def test_training_status_not_found(client: TestClient):
    """Test getting status for non-existent job"""
    response = client.get("/api/v1/training/status/nonexistent-job-id")
    assert response.status_code == 404


def test_list_training_jobs(client: TestClient):
    """Test listing training jobs"""
    response = client.get("/api/v1/training/list")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "jobs" in data
    assert isinstance(data["jobs"], list)


def test_cancel_nonexistent_job(client: TestClient):
    """Test canceling non-existent job"""
    response = client.delete("/api/v1/training/cancel/nonexistent-job-id")
    assert response.status_code == 404