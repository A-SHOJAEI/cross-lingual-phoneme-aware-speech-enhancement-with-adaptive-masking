"""Analysis and visualization utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")


def plot_training_curves(
    history: Dict,
    output_path: str,
) -> None:
    """Plot training and validation loss curves.

    Args:
        history: Training history dictionary.
        output_path: Path to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning rate
    axes[1].plot(history["learning_rate"], linewidth=2, color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved training curves to {output_path}")


def plot_spectrograms(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    output_path: str,
) -> None:
    """Plot spectrograms for comparison.

    Args:
        clean: Clean audio waveform.
        noisy: Noisy audio waveform.
        enhanced: Enhanced audio waveform.
        sample_rate: Sample rate in Hz.
        output_path: Path to save the plot.
    """
    import librosa
    import librosa.display

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Compute spectrograms
    D_clean = librosa.amplitude_to_db(
        np.abs(librosa.stft(clean)), ref=np.max
    )
    D_noisy = librosa.amplitude_to_db(
        np.abs(librosa.stft(noisy)), ref=np.max
    )
    D_enhanced = librosa.amplitude_to_db(
        np.abs(librosa.stft(enhanced)), ref=np.max
    )

    # Plot
    img1 = librosa.display.specshow(
        D_clean, y_axis="hz", x_axis="time", sr=sample_rate, ax=axes[0]
    )
    axes[0].set_title("Clean Audio")
    axes[0].label_outer()

    img2 = librosa.display.specshow(
        D_noisy, y_axis="hz", x_axis="time", sr=sample_rate, ax=axes[1]
    )
    axes[1].set_title("Noisy Audio")
    axes[1].label_outer()

    img3 = librosa.display.specshow(
        D_enhanced, y_axis="hz", x_axis="time", sr=sample_rate, ax=axes[2]
    )
    axes[2].set_title("Enhanced Audio")

    # Add colorbar
    fig.colorbar(img3, ax=axes, format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved spectrogram comparison to {output_path}")


def save_audio_samples(
    samples: List[Dict[str, np.ndarray]],
    output_dir: str,
    sample_rate: int = 16000,
) -> None:
    """Save audio samples for listening tests.

    Args:
        samples: List of dictionaries with 'clean', 'noisy', 'enhanced' keys.
        output_dir: Directory to save audio files.
        sample_rate: Sample rate in Hz.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        # Save clean
        clean_path = output_path / f"sample_{i:03d}_clean.wav"
        sf.write(clean_path, sample["clean"], sample_rate)

        # Save noisy
        noisy_path = output_path / f"sample_{i:03d}_noisy.wav"
        sf.write(noisy_path, sample["noisy"], sample_rate)

        # Save enhanced
        enhanced_path = output_path / f"sample_{i:03d}_enhanced.wav"
        sf.write(enhanced_path, sample["enhanced"], sample_rate)

    logger.info(f"Saved {len(samples)} audio samples to {output_dir}")


def plot_metric_comparison(
    results: Dict[str, Dict],
    output_path: str,
) -> None:
    """Plot metric comparison across different configurations.

    Args:
        results: Dictionary mapping config names to metric dictionaries.
        output_path: Path to save the plot.
    """
    metrics = ["pesq", "stoi", "si_sdr", "phoneme_preservation_rate"]
    config_names = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = [results[config].get(metric, 0) for config in config_names]
        stds = [results[config].get(f"{metric}_std", 0) for config in config_names]

        x = np.arange(len(config_names))
        axes[i].bar(x, values, yerr=stds, capsize=5, alpha=0.7)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(config_names, rotation=45, ha="right")
        axes[i].set_ylabel(metric.upper().replace("_", " "))
        axes[i].set_title(f"{metric.upper()} Comparison")
        axes[i].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved metric comparison to {output_path}")


def generate_evaluation_report(
    results: Dict,
    output_path: str,
) -> None:
    """Generate a comprehensive evaluation report.

    Args:
        results: Evaluation results dictionary.
        output_path: Path to save the report.
    """
    with open(output_path, "w") as f:
        f.write("# Speech Enhancement Evaluation Report\n\n")

        # Overall metrics
        f.write("## Overall Metrics\n\n")
        f.write("| Metric | Value | Std Dev |\n")
        f.write("|--------|-------|----------|\n")

        for key, value in results.items():
            if not key.endswith("_std"):
                std_key = f"{key}_std"
                std_value = results.get(std_key, 0.0)
                f.write(f"| {key} | {value:.4f} | {std_value:.4f} |\n")

        f.write("\n")

        # Target metrics comparison
        f.write("## Target Metrics Comparison\n\n")
        targets = {
            "pesq": 3.5,
            "stoi": 0.88,
            "phoneme_preservation_rate": 0.92,
        }

        f.write("| Metric | Target | Achieved | Status |\n")
        f.write("|--------|--------|----------|--------|\n")

        for metric, target in targets.items():
            achieved = results.get(metric, 0.0)
            status = "PASS" if achieved >= target else "FAIL"
            f.write(f"| {metric} | {target:.4f} | {achieved:.4f} | {status} |\n")

    logger.info(f"Saved evaluation report to {output_path}")
