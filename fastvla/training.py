"""
Training loop for FastVLA models.
Includes training, evaluation, and checkpointing utilities.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Callable
from tqdm import tqdm
import os
from pathlib import Path
import json
from .optimization import get_8bit_optimizer, setup_mixed_precision_training


class FastVLATrainer:
    """
    Trainer for FastVLA models with Unsloth-style optimizations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_8bit_optimizer: bool = True,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        output_dir: str = "./checkpoints",
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader (optional)
            optimizer: Optimizer (optional, will create 8-bit optimizer if not provided)
            lr_scheduler: Learning rate scheduler (optional)
            use_8bit_optimizer: Whether to use 8-bit optimizer
            use_mixed_precision: Whether to use mixed precision training
            gradient_accumulation_steps: Number of gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on
            output_dir: Output directory for checkpoints
            save_steps: Steps between checkpoints
            eval_steps: Steps between evaluations
            logging_steps: Steps between logging
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        if optimizer is None:
            if use_8bit_optimizer:
                self.optimizer = get_8bit_optimizer(model, learning_rate=5e-5)
            else:
                self.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=5e-5,
                    weight_decay=0.01,
                )
        else:
            self.optimizer = optimizer
        
        self.lr_scheduler = lr_scheduler
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.training_history = []
        
        # Setup mixed precision scaler
        if use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast() if self.use_mixed_precision else torch.no_grad():
            action_preds, loss = self.model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            self.optimizer.zero_grad()
        
        return {
            "loss": loss.item() * self.gradient_accumulation_steps,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                with torch.cuda.amp.autocast() if self.use_mixed_precision else torch.no_grad():
                    action_preds, loss = self.model(
                        pixel_values=batch["pixel_values"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch.get("labels"),
                    )
                
                total_loss += loss.item()
                num_samples += batch["pixel_values"].size(0)
        
        avg_loss = total_loss / len(self.eval_dataloader) if len(self.eval_dataloader) > 0 else 0.0
        
        return {
            "eval_loss": avg_loss,
            "eval_samples": num_samples,
        }
    
    def save_checkpoint(self, step: Optional[int] = None):
        """
        Save a checkpoint.
        
        Args:
            step: Step number (uses global_step if not provided)
        """
        if step is None:
            step = self.global_step
        
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), checkpoint_dir / "pytorch_model.bin")
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "training_history": self.training_history,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def train(self, num_epochs: int = 1, max_steps: Optional[int] = None):
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            max_steps: Maximum number of training steps (optional)
        """
        self.model.train()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            for batch in progress_bar:
                # Training step
                metrics = self.train_step(batch)
                epoch_loss += metrics["loss"]
                num_batches += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{metrics['learning_rate']:.2e}",
                    })
                    self.training_history.append({
                        "step": self.global_step,
                        "loss": avg_loss,
                        "learning_rate": metrics["learning_rate"],
                    })
                
                # Evaluation
                if self.eval_dataloader is not None and self.global_step % self.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    print(f"\nStep {self.global_step} - Eval Loss: {eval_metrics.get('eval_loss', 0.0):.4f}")
                    self.training_history[-1].update(eval_metrics)
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint()
                
                # Check max steps
                if max_steps is not None and self.global_step >= max_steps:
                    break
            
            # Final checkpoint at end of epoch
            self.save_checkpoint()
            
            if max_steps is not None and self.global_step >= max_steps:
                break
        
        print("Training completed!")
        return self.training_history

