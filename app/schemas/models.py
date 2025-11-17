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
