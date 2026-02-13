"""Tests for training infrastructure."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.models import (
    PhonemeAwareEnhancer,
)
from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.training import Trainer


class TestTrainer:
    """Tests for trainer class."""

    @pytest.fixture
    def dummy_dataloader(self):
        """Create a dummy dataloader for testing."""
        batch_size = 2
        n_mels = 128
        time_steps = 50
        phoneme_dim = 39

        # Create dummy data
        clean_mel = torch.randn(10, n_mels, time_steps)
        noisy_mel = torch.randn(10, n_mels, time_steps)
        clean_audio = torch.randn(10, 16000)
        noisy_audio = torch.randn(10, 16000)
        phoneme_features = torch.randn(10, phoneme_dim, time_steps)
        language = torch.randint(0, 4, (10,))

        dataset = TensorDataset(
            clean_mel, noisy_mel, clean_audio, noisy_audio, phoneme_features, language
        )

        # Custom collate function
        def collate_fn(batch):
            return {
                "clean_mel": torch.stack([item[0] for item in batch]),
                "noisy_mel": torch.stack([item[1] for item in batch]),
                "clean_audio": torch.stack([item[2] for item in batch]),
                "noisy_audio": torch.stack([item[3] for item in batch]),
                "phoneme_features": torch.stack([item[4] for item in batch]),
                "language": torch.stack([item[5] for item in batch]),
            }

        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    def test_trainer_creation(self, model_config, dummy_dataloader):
        """Test trainer creation."""
        model = PhonemeAwareEnhancer(**model_config)

        config = {
            "training": {
                "num_epochs": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip": 5.0,
                "mixed_precision": False,  # Disable for testing
                "accumulation_steps": 1,
                "scheduler": {"type": "cosine", "warmup_epochs": 1, "min_lr": 0.00001},
                "early_stopping": {"patience": 5, "min_delta": 0.001},
                "loss_weights": {
                    "reconstruction": 1.0,
                    "perceptual": 0.5,
                    "contrastive": 0.3,
                    "phoneme_preservation": 0.4,
                },
            },
            "data": {"sample_rate": 16000},
            "model": {"temperature": 0.07},
        }

        trainer = Trainer(
            model=model,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            config=config,
            device=torch.device("cpu"),
            checkpoint_dir="./test_checkpoints",
        )

        assert trainer is not None
        assert trainer.num_epochs == 2

    def test_trainer_train_epoch(self, model_config, dummy_dataloader):
        """Test single training epoch."""
        model = PhonemeAwareEnhancer(**model_config)

        config = {
            "training": {
                "num_epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip": 5.0,
                "mixed_precision": False,
                "accumulation_steps": 1,
                "scheduler": {"type": "cosine", "warmup_epochs": 0, "min_lr": 0.00001},
                "early_stopping": {"patience": 5, "min_delta": 0.001},
                "loss_weights": {
                    "reconstruction": 1.0,
                    "perceptual": 0.0,  # Disable for speed
                    "contrastive": 0.0,
                    "phoneme_preservation": 0.0,
                },
            },
            "data": {"sample_rate": 16000},
            "model": {"temperature": 0.07},
        }

        trainer = Trainer(
            model=model,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            config=config,
            device=torch.device("cpu"),
            checkpoint_dir="./test_checkpoints",
        )

        train_loss = trainer.train_epoch(epoch=0)

        assert isinstance(train_loss, float)
        assert train_loss >= 0

    def test_trainer_validate(self, model_config, dummy_dataloader):
        """Test validation."""
        model = PhonemeAwareEnhancer(**model_config)

        config = {
            "training": {
                "num_epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip": 5.0,
                "mixed_precision": False,
                "accumulation_steps": 1,
                "scheduler": {"type": "cosine", "warmup_epochs": 0, "min_lr": 0.00001},
                "early_stopping": {"patience": 5, "min_delta": 0.001},
                "loss_weights": {
                    "reconstruction": 1.0,
                    "perceptual": 0.0,
                    "contrastive": 0.0,
                    "phoneme_preservation": 0.0,
                },
            },
            "data": {"sample_rate": 16000},
            "model": {"temperature": 0.07},
        }

        trainer = Trainer(
            model=model,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            config=config,
            device=torch.device("cpu"),
            checkpoint_dir="./test_checkpoints",
        )

        val_loss = trainer.validate()

        assert isinstance(val_loss, float)
        assert val_loss >= 0

    def test_trainer_loss_computation(self, model_config, dummy_dataloader):
        """Test loss computation."""
        model = PhonemeAwareEnhancer(**model_config)

        config = {
            "training": {
                "num_epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip": 5.0,
                "mixed_precision": False,
                "accumulation_steps": 1,
                "scheduler": {"type": "cosine", "warmup_epochs": 0, "min_lr": 0.00001},
                "early_stopping": {"patience": 5, "min_delta": 0.001},
                "loss_weights": {
                    "reconstruction": 1.0,
                    "perceptual": 0.5,
                    "contrastive": 0.3,
                    "phoneme_preservation": 0.4,
                },
            },
            "data": {"sample_rate": 16000},
            "model": {"temperature": 0.07},
        }

        trainer = Trainer(
            model=model,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            config=config,
            device=torch.device("cpu"),
            checkpoint_dir="./test_checkpoints",
        )

        # Get a batch
        batch = next(iter(dummy_dataloader))

        # Forward pass
        noisy_mel = batch["noisy_mel"]
        phoneme_features = batch["phoneme_features"]
        language = batch["language"]

        outputs = model(noisy_mel, phoneme_features, language)

        # Compute loss
        losses = trainer.compute_loss(batch, outputs)

        assert "total" in losses
        assert "reconstruction" in losses
        assert losses["total"].item() >= 0
