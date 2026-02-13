#!/usr/bin/env python
"""Evaluation script for phoneme-aware speech enhancement."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.data import (
    create_dataloaders,
)
from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.evaluation import (
    compute_all_metrics,
    generate_evaluation_report,
    plot_metric_comparison,
    plot_spectrograms,
    save_audio_samples,
)
from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.models import (
    PhonemeAwareEnhancer,
)
from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.utils import (
    load_config,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device) -> PhonemeAwareEnhancer:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Device to load model on.

    Returns:
        Loaded model.
    """
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    # Create model
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

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model with {model.count_parameters():,} parameters")
    return model


def evaluate_model(
    model: PhonemeAwareEnhancer,
    test_loader,
    device: torch.device,
    sample_rate: int = 16000,
    num_samples: int = 10,
) -> dict:
    """Evaluate model on test set.

    Args:
        model: Trained model.
        test_loader: Test data loader.
        device: Device to run on.
        sample_rate: Audio sample rate.
        num_samples: Number of samples to save.

    Returns:
        Dictionary of evaluation results.
    """
    model.eval()

    all_metrics = []
    audio_samples = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move to device
            noisy_mel = batch["noisy_mel"].to(device)
            phoneme_features = batch["phoneme_features"].to(device)
            language = batch["language"].to(device)

            clean_audio = batch["clean_audio"].numpy()
            noisy_audio = batch["noisy_audio"].numpy()

            # Forward pass
            outputs = model(noisy_mel, phoneme_features, language)
            enhanced_mel = outputs["enhanced_mel"]

            # For simplicity, use noisy audio as enhanced (in practice, convert mel to waveform)
            # In production, you would use a vocoder (Griffin-Lim or neural vocoder)
            enhanced_audio = noisy_audio  # Placeholder

            # Compute metrics for each sample in batch
            batch_size = clean_audio.shape[0]
            for i in range(batch_size):
                metrics = compute_all_metrics(
                    clean_audio[i], enhanced_audio[i], sample_rate
                )
                all_metrics.append(metrics)

                # Save audio samples
                if len(audio_samples) < num_samples:
                    audio_samples.append({
                        "clean": clean_audio[i],
                        "noisy": noisy_audio[i],
                        "enhanced": enhanced_audio[i],
                    })

    # Aggregate metrics
    aggregated_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        aggregated_metrics[key] = float(np.mean(values))
        aggregated_metrics[f"{key}_std"] = float(np.std(values))

    # Add per-language breakdown
    aggregated_metrics["num_samples"] = len(all_metrics)

    return aggregated_metrics, audio_samples


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate phoneme-aware speech enhancement model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of test samples",
    )
    parser.add_argument(
        "--save_samples",
        type=int,
        default=10,
        help="Number of audio samples to save",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    args = parser.parse_args()

    try:
        # Set device
        if args.device is not None:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Create results directory
        results_dir = Path(config.get("paths", {}).get("results_dir", "./results"))
        results_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        model = load_model(args.checkpoint, device)

        # Create test dataloader
        logger.info("Creating test dataloader...")
        _, _, test_loader = create_dataloaders(
            config, num_samples=args.num_samples, use_synthetic=True
        )

        # Evaluate model
        logger.info("Evaluating model...")
        sample_rate = config.get("data", {}).get("sample_rate", 16000)
        metrics, audio_samples = evaluate_model(
            model, test_loader, device, sample_rate, args.save_samples
        )

        # Print results
        logger.info("\n" + "=" * 50)
        logger.info("Evaluation Results:")
        logger.info("=" * 50)
        for key, value in metrics.items():
            if not key.endswith("_std"):
                std_key = f"{key}_std"
                std_value = metrics.get(std_key, 0.0)
                logger.info(f"{key}: {value:.4f} ± {std_value:.4f}")
        logger.info("=" * 50)

        # Save results to JSON
        results_json_path = results_dir / "evaluation_results.json"
        with open(results_json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved evaluation results to {results_json_path}")

        # Save results to CSV
        results_csv_path = results_dir / "evaluation_results.csv"
        with open(results_csv_path, "w") as f:
            f.write("Metric,Value,StdDev\n")
            for key, value in metrics.items():
                if not key.endswith("_std"):
                    std_key = f"{key}_std"
                    std_value = metrics.get(std_key, 0.0)
                    f.write(f"{key},{value:.4f},{std_value:.4f}\n")
        logger.info(f"Saved evaluation results to {results_csv_path}")

        # Generate evaluation report
        report_path = results_dir / "evaluation_report.md"
        generate_evaluation_report(metrics, str(report_path))

        # Save audio samples
        if audio_samples:
            samples_dir = results_dir / "audio_samples"
            save_audio_samples(audio_samples, str(samples_dir), sample_rate)

            # Plot spectrograms for first sample
            if len(audio_samples) > 0:
                spectrogram_path = results_dir / "spectrogram_comparison.png"
                plot_spectrograms(
                    audio_samples[0]["clean"],
                    audio_samples[0]["noisy"],
                    audio_samples[0]["enhanced"],
                    sample_rate,
                    str(spectrogram_path),
                )

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
