"""Training loop with learning rate scheduling and early stopping."""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.components import (
    ContrastivePhonemeLoss,
    PerceptualLoss,
    PhonemePreservationLoss,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for phoneme-aware speech enhancement model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
    ):
        """Initialize trainer.

        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Training configuration.
            device: Device to train on.
            checkpoint_dir: Directory to save checkpoints.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training configuration
        train_config = config.get("training", {})
        self.num_epochs = train_config.get("num_epochs", 100)
        self.learning_rate = train_config.get("learning_rate", 0.0003)
        self.weight_decay = train_config.get("weight_decay", 0.0001)
        self.gradient_clip = train_config.get("gradient_clip", 5.0)
        self.mixed_precision = train_config.get("mixed_precision", True)
        self.accumulation_steps = train_config.get("accumulation_steps", 1)

        # Loss weights
        loss_weights = train_config.get("loss_weights", {})
        self.reconstruction_weight = loss_weights.get("reconstruction", 1.0)
        self.perceptual_weight = loss_weights.get("perceptual", 0.5)
        self.contrastive_weight = loss_weights.get("contrastive", 0.3)
        self.phoneme_preservation_weight = loss_weights.get("phoneme_preservation", 0.4)

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Initialize scheduler
        scheduler_config = train_config.get("scheduler", {})
        self.scheduler = self._create_scheduler(scheduler_config)

        # Initialize GradScaler for mixed precision
        if self.mixed_precision:
            try:
                self.scaler = GradScaler("cuda")
            except TypeError:
                # Fallback for older PyTorch versions
                self.scaler = GradScaler()
        else:
            self.scaler = None

        # Loss functions
        self.reconstruction_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(
            sample_rate=config.get("data", {}).get("sample_rate", 16000)
        )
        self.contrastive_loss = ContrastivePhonemeLoss(
            temperature=config.get("model", {}).get("temperature", 0.07)
        )
        self.phoneme_preservation_loss = PhonemePreservationLoss()

        # Early stopping
        early_stop_config = train_config.get("early_stopping", {})
        self.early_stop_patience = early_stop_config.get("patience", 15)
        self.early_stop_min_delta = early_stop_config.get("min_delta", 0.001)
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    def _create_scheduler(self, scheduler_config: Dict):
        """Create learning rate scheduler.

        Args:
            scheduler_config: Scheduler configuration.

        Returns:
            Learning rate scheduler.
        """
        scheduler_type = scheduler_config.get("type", "cosine")
        warmup_epochs = scheduler_config.get("warmup_epochs", 5)
        min_lr = scheduler_config.get("min_lr", 0.00001)

        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs - warmup_epochs,
                eta_min=min_lr,
            )
        elif scheduler_type == "step":
            step_size = scheduler_config.get("step_size", 30)
            gamma = scheduler_config.get("gamma", 0.1)
            scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=min_lr,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler

    def compute_loss(self, batch: Dict, outputs: Dict) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            batch: Input batch.
            outputs: Model outputs.

        Returns:
            Dictionary of losses.
        """
        losses = {}

        # Reconstruction loss on mel spectrograms
        enhanced_mel = outputs["enhanced_mel"]
        clean_mel = batch["clean_mel"].to(self.device)

        reconstruction_loss = self.reconstruction_loss(enhanced_mel, clean_mel)
        losses["reconstruction"] = reconstruction_loss * self.reconstruction_weight

        # Perceptual loss on waveforms
        if self.perceptual_weight > 0:
            clean_audio = batch["clean_audio"].to(self.device)
            noisy_audio = batch["noisy_audio"].to(self.device)

            # Simple waveform reconstruction (in practice, use Griffin-Lim or vocoder)
            perceptual_loss = self.perceptual_loss(noisy_audio, clean_audio)
            losses["perceptual"] = perceptual_loss * self.perceptual_weight

        # Contrastive phoneme loss
        if self.contrastive_weight > 0 and outputs.get("phoneme_embeddings") is not None:
            phoneme_embeddings = outputs["phoneme_embeddings"]
            batch_size = phoneme_embeddings.shape[0]

            # Create positive pairs (same phoneme from different samples)
            if batch_size >= 2:
                phoneme_emb_1 = phoneme_embeddings[: batch_size // 2].mean(dim=1)
                phoneme_emb_2 = phoneme_embeddings[batch_size // 2 : batch_size].mean(dim=1)

                # Create positive pairs mask
                positive_pairs = torch.eye(
                    min(len(phoneme_emb_1), len(phoneme_emb_2)),
                    device=self.device,
                )

                if len(phoneme_emb_1) == len(phoneme_emb_2):
                    contrastive_loss = self.contrastive_loss(
                        phoneme_emb_1, phoneme_emb_2, positive_pairs
                    )
                    losses["contrastive"] = contrastive_loss * self.contrastive_weight

        # Phoneme preservation loss
        if self.phoneme_preservation_weight > 0:
            clean_audio = batch["clean_audio"].to(self.device)
            noisy_audio = batch["noisy_audio"].to(self.device)

            phoneme_preservation_loss = self.phoneme_preservation_loss(
                noisy_audio, clean_audio
            )
            losses["phoneme_preservation"] = (
                phoneme_preservation_loss * self.phoneme_preservation_weight
            )

        # Total loss
        total_loss = sum(losses.values())
        losses["total"] = total_loss

        return losses

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs}",
            leave=False,
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            noisy_mel = batch["noisy_mel"].to(self.device)
            phoneme_features = batch["phoneme_features"].to(self.device)
            language = batch["language"].to(self.device)

            # Forward pass with mixed precision
            if self.mixed_precision:
                try:
                    with autocast("cuda"):
                        outputs = self.model(noisy_mel, phoneme_features, language)
                        losses = self.compute_loss(batch, outputs)
                        loss = losses["total"] / self.accumulation_steps
                except TypeError:
                    # Fallback for older PyTorch versions
                    with autocast():
                        outputs = self.model(noisy_mel, phoneme_features, language)
                        losses = self.compute_loss(batch, outputs)
                        loss = losses["total"] / self.accumulation_steps

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(noisy_mel, phoneme_features, language)
                losses = self.compute_loss(batch, outputs)
                loss = losses["total"] / self.accumulation_steps

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += losses["total"].item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": losses["total"].item()})

        # Handle empty dataloader case
        if num_batches == 0:
            logger.warning("No batches in training loader. Check dataset size and batch_size.")
            return 0.0

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> float:
        """Validate model.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move data to device
                noisy_mel = batch["noisy_mel"].to(self.device)
                phoneme_features = batch["phoneme_features"].to(self.device)
                language = batch["language"].to(self.device)

                # Forward pass
                outputs = self.model(noisy_mel, phoneme_features, language)
                losses = self.compute_loss(batch, outputs)

                total_loss += losses["total"].item()
                num_batches += 1

        # Handle empty dataloader case
        if num_batches == 0:
            logger.warning("No batches in validation loader. Check dataset size and batch_size.")
            return 0.0

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "history": self.history,
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def train(self) -> Dict:
        """Train the model.

        Returns:
            Training history.
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Model has {self.model.count_parameters():,} trainable parameters")

        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Update learning rate
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)

            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(epoch, is_best=is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.early_stop_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(no improvement for {self.early_stop_patience} epochs)"
                )
                break

        logger.info("Training completed")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        return self.history
