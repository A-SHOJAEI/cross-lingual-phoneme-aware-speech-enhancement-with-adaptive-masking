#!/usr/bin/env python
"""Training script for phoneme-aware speech enhancement."""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.data import (
    create_dataloaders,
)
from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.evaluation import (
    plot_training_curves,
)
from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.models import (
    PhonemeAwareEnhancer,
)
from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.training import Trainer
from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.utils import (
    load_config,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train phoneme-aware speech enhancement model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    args = parser.parse_args()

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Set random seed
        seed = config.get("seed", 42)
        set_seed(seed)

        # Set device
        if args.device is not None:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Create directories
        paths = config.get("paths", {})
        model_dir = Path(paths.get("model_dir", "./models"))
        checkpoint_dir = Path(paths.get("checkpoint_dir", "./checkpoints"))
        results_dir = Path(paths.get("results_dir", "./results"))

        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MLflow (wrapped in try/except)
        try:
            import mlflow

            mlflow_config = config.get("mlflow", {})
            mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "./mlruns"))
            mlflow.set_experiment(mlflow_config.get("experiment_name", "default"))

            # Start MLflow run
            mlflow.start_run(run_name=mlflow_config.get("run_name", "default"))
            mlflow.log_params({"seed": seed, "device": str(device)})
            mlflow.log_params(config.get("training", {}))
            logger.info("MLflow tracking enabled")
            mlflow_enabled = True
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")
            mlflow_enabled = False

        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            config, num_samples=args.num_samples, use_synthetic=True
        )

        # Create model
        logger.info("Creating model...")
        model_config = config.get("model", {})
        data_config = config.get("data", {})

        model = PhonemeAwareEnhancer(
            n_mels=data_config.get("n_mels", 128),
            encoder_dim=model_config.get("encoder_dim", 256),
            decoder_dim=model_config.get("decoder_dim", 256),
            phoneme_embedding_dim=model_config.get("phoneme_embedding_dim", 128),
            num_phonemes=model_config.get("num_phonemes", 128),
            num_attention_heads=model_config.get("num_attention_heads", 8),
            num_encoder_layers=model_config.get("num_encoder_layers", 6),
            num_decoder_layers=model_config.get("num_decoder_layers", 6),
            dropout=model_config.get("dropout", 0.1),
            use_contrastive_alignment=model_config.get("use_contrastive_alignment", True),
            cross_lingual_transfer=model_config.get("cross_lingual_transfer", True),
        )

        logger.info(f"Model has {model.count_parameters():,} trainable parameters")

        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=str(checkpoint_dir),
        )

        # Train model
        logger.info("Starting training...")
        history = trainer.train()

        # Save training history
        history_path = results_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

        # Plot training curves
        curves_path = results_dir / "training_curves.png"
        plot_training_curves(history, str(curves_path))

        # Save final model
        final_model_path = model_dir / "final_model.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
        }, final_model_path)
        logger.info(f"Saved final model to {final_model_path}")

        # Log results to MLflow
        if mlflow_enabled:
            try:
                mlflow.log_metric("best_val_loss", trainer.best_val_loss)
                mlflow.log_artifact(str(history_path))
                mlflow.log_artifact(str(curves_path))
                mlflow.log_artifact(str(final_model_path))
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Error logging to MLflow: {e}")

        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if mlflow_enabled:
            try:
                mlflow.end_run(status="FAILED")
            except:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
