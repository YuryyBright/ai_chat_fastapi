# app/core/training.py
"""
Training manager for model fine-tuning
"""

import os
import time
from typing import Dict, Optional, List
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from app.schemas.training import TrainingRequest, TrainingStatus
from app.core.exceptions import TrainingError
import logging
import torch

logger = logging.getLogger(__name__)


class TrainingManager:
    """
    Manages model training jobs
    """
    
    def __init__(self):
        self.jobs: Dict[str, TrainingStatus] = {}
        self.active_trainers: Dict[str, Trainer] = {}
    
    def update_status(self, job_id: str, status: TrainingStatus) -> None:
        """Update status of a training job"""
        self.jobs[job_id] = status
    
    def get_status(self, job_id: str) -> Optional[TrainingStatus]:
        """Get status of a training job"""
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[TrainingStatus]:
        """List all training jobs"""
        return list(self.jobs.values())
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        if job_id not in self.jobs:
            return False
        
        status = self.jobs[job_id]
        if status.status in ["completed", "failed"]:
            return False
        
        # Update status
        status.status = "failed"
        status.error = "Job cancelled by user"
        status.completed_at = datetime.utcnow()
        self.update_status(job_id, status)
        
        return True
    
    async def train_model(
        self,
        job_id: str,
        request: TrainingRequest,
        provider_manager
    ) -> None:
        """
        Execute model training
        
        Args:
            job_id: Unique job identifier
            request: Training request parameters
            provider_manager: Provider manager instance
        """
        status = self.get_status(job_id)
        status.status = "running"
        status.started_at = datetime.utcnow()
        status.total_epochs = request.epochs
        self.update_status(job_id, status)
        
        try:
            logger.info(f"Starting training for job {job_id}")
            
            # Get provider
            if request.provider != "huggingface":
                raise TrainingError("Only HuggingFace provider supports training")
            
            provider = provider_manager.get_provider("huggingface")
            
            # Load base model and tokenizer
            logger.info(f"Loading base model: {request.base_model}")
            tokenizer = AutoTokenizer.from_pretrained(
                request.base_model,
                cache_dir=provider.cache_dir
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                request.base_model,
                cache_dir=provider.cache_dir
            )
            
            # Prepare dataset
            logger.info("Preparing training dataset")
            dataset_dict = {"text": request.dataset.texts}
            if request.dataset.labels:
                dataset_dict["labels"] = request.dataset.labels
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Tokenize dataset
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Setup training arguments
            output_dir = os.path.join(
                provider.config.get("output_dir", "./models/fine-tuned"),
                request.output_name
            )
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=request.epochs,
                per_device_train_batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                warmup_steps=request.warmup_steps,
                weight_decay=request.weight_decay,
                save_steps=request.save_steps,
                logging_steps=100,
                evaluation_strategy="no",
                save_strategy="steps",
                load_best_model_at_end=False,
                report_to=[]
            )
            
            # Create trainer
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )
            
            self.active_trainers[job_id] = trainer
            
            # Custom callback for progress updates
            class ProgressCallback:
                def __init__(self, manager, job_id, total_epochs):
                    self.manager = manager
                    self.job_id = job_id
                    self.total_epochs = total_epochs
                
                def on_epoch_end(self, args, state, control, **kwargs):
                    status = self.manager.get_status(self.job_id)
                    status.current_epoch = state.epoch
                    status.progress = (state.epoch / self.total_epochs) * 100
                    if hasattr(state, 'log_history') and state.log_history:
                        status.loss = state.log_history[-1].get('loss')
                    self.manager.update_status(self.job_id, status)
            
            # Train model
            logger.info(f"Starting training for {request.epochs} epochs")
            trainer.train()
            
            # Save final model
            logger.info(f"Saving model to {output_dir}")
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Update status to completed
            status.status = "completed"
            status.progress = 100.0
            status.model_path = output_dir
            status.completed_at = datetime.utcnow()
            self.update_status(job_id, status)
            
            logger.info(f"Training completed successfully for job {job_id}")
            
            # Cleanup
            del self.active_trainers[job_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Training failed for job {job_id}: {e}")
            status.status = "failed"
            status.error = str(e)
            status.completed_at = datetime.utcnow()
            self.update_status(job_id, status)
            
            # Cleanup
            if job_id in self.active_trainers:
                del self.active_trainers[job_id]