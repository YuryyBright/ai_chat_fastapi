# app/api/routes/training.py
"""
API routes for model training and fine-tuning
"""

from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from app.schemas.training import TrainingRequest, TrainingStatus
from app.core.training import TrainingManager
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Global training manager instance
training_manager = TrainingManager()


@router.post("/start", response_model=TrainingStatus)
async def start_training(
    training_request: TrainingRequest,
    background_tasks: BackgroundTasks,
    request: Request
) -> TrainingStatus:
    """
    Start a new model training/fine-tuning job
    
    Training runs in the background. Use the job_id to check status.
    
    - **base_model**: Base model to fine-tune
    - **provider**: Training provider (currently only 'huggingface')
    - **dataset**: Training dataset with texts and optional labels
    - **output_name**: Name for the fine-tuned model
    - **epochs**: Number of training epochs
    - **batch_size**: Training batch size
    - **learning_rate**: Learning rate
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create initial status
    status = TrainingStatus(
        job_id=job_id,
        status="pending",
        progress=0.0
    )
    
    # Store initial status
    training_manager.update_status(job_id, status)
    
    # Start training in background
    background_tasks.add_task(
        training_manager.train_model,
        job_id=job_id,
        request=training_request,
        provider_manager=request.app.state.provider_manager
    )
    
    logger.info(f"Training job started: {job_id} for model {training_request.output_name}")
    
    return status


@router.get("/status/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str) -> TrainingStatus:
    """
    Get status of a training job
    
    - **job_id**: Training job ID returned from /start endpoint
    """
    status = training_manager.get_status(job_id)
    
    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Training job not found: {job_id}"
        )
    
    return status


@router.get("/list")
async def list_training_jobs():
    """
    List all training jobs
    
    Returns a list of all training jobs and their statuses
    """
    jobs = training_manager.list_jobs()
    return {
        "total": len(jobs),
        "jobs": jobs
    }


@router.delete("/cancel/{job_id}")
async def cancel_training(job_id: str):
    """
    Cancel a running training job
    
    - **job_id**: Training job ID to cancel
    """
    success = training_manager.cancel_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Training job not found or already completed: {job_id}"
        )
    
    logger.info(f"Training job cancelled: {job_id}")
    
    return {
        "message": "Training job cancelled",
        "job_id": job_id
    }
