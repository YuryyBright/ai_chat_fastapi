# app/schemas/training.py
"""
Pydantic schemas for model training
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


class TrainingDataset(BaseModel):
    """Schema for training dataset"""
    
    texts: List[str] = Field(
        ...,
        min_length=1,
        description="Training texts"
    )
    
    labels: Optional[List[str]] = Field(
        default=None,
        description="Training labels (for supervised learning)"
    )
    
    @validator("labels")
    def validate_labels(cls, v, values):
        """Ensure labels match texts length if provided"""
        if v is not None and "texts" in values:
            if len(v) != len(values["texts"]):
                raise ValueError("Labels and texts must have same length")
        return v


class TrainingRequest(BaseModel):
    """Schema for model training request"""
    
    base_model: str = Field(
        ...,
        description="Base model to fine-tune"
    )
    
    provider: Literal["huggingface"] = Field(
        default="huggingface",
        description="Provider for training (currently only HuggingFace)"
    )
    
    dataset: TrainingDataset = Field(
        ...,
        description="Training dataset"
    )
    
    output_name: str = Field(
        ...,
        description="Name for the fine-tuned model"
    )
    
    epochs: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of training epochs"
    )
    
    batch_size: int = Field(
        default=8,
        ge=1,
        le=128,
        description="Training batch size"
    )
    
    learning_rate: float = Field(
        default=2e-5,
        gt=0.0,
        lt=1.0,
        description="Learning rate"
    )
    
    warmup_steps: int = Field(
        default=500,
        ge=0,
        description="Number of warmup steps"
    )
    
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Weight decay"
    )
    
    save_steps: int = Field(
        default=1000,
        ge=1,
        description="Save checkpoint every N steps"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "base_model": "gpt2",
                "provider": "huggingface",
                "dataset": {
                    "texts": ["Sample text 1", "Sample text 2"]
                },
                "output_name": "my-fine-tuned-model",
                "epochs": 3,
                "batch_size": 8,
                "learning_rate": 2e-5
            }
        }


class TrainingStatus(BaseModel):
    """Schema for training status"""
    
    job_id: str = Field(
        ...,
        description="Training job ID"
    )
    
    status: Literal["pending", "running", "completed", "failed"] = Field(
        ...,
        description="Training status"
    )
    
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Training progress percentage"
    )
    
    current_epoch: Optional[int] = Field(
        default=None,
        description="Current training epoch"
    )
    
    total_epochs: Optional[int] = Field(
        default=None,
        description="Total training epochs"
    )
    
    loss: Optional[float] = Field(
        default=None,
        description="Current training loss"
    )
    
    model_path: Optional[str] = Field(
        default=None,
        description="Path to saved model (if completed)"
    )
    
    started_at: Optional[datetime] = Field(
        default=None,
        description="Training start time"
    )
    
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Training completion time"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message (if failed)"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional training metadata"
    )