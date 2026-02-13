"""Evaluation metrics for speech enhancement."""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_pesq(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000,
) -> float:
    """Compute PESQ (Perceptual Evaluation of Speech Quality) score.

    Args:
        reference: Reference (clean) audio.
        degraded: Degraded (enhanced/noisy) audio.
        sample_rate: Sample rate in Hz.

    Returns:
        PESQ score (higher is better, range: -0.5 to 4.5).
    """
    try:
        from pesq import pesq

        # PESQ requires specific sample rates
        if sample_rate not in [8000, 16000]:
            logger.warning(f"PESQ requires 8kHz or 16kHz, got {sample_rate}Hz")
            return 0.0

        mode = "wb" if sample_rate == 16000 else "nb"
        score = pesq(sample_rate, reference, degraded, mode)
        return float(score)
    except Exception as e:
        logger.warning(f"Error computing PESQ: {e}")
        return 0.0


def compute_stoi(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int = 16000,
) -> float:
    """Compute STOI (Short-Time Objective Intelligibility) score.

    Args:
        reference: Reference (clean) audio.
        degraded: Degraded (enhanced/noisy) audio.
        sample_rate: Sample rate in Hz.

    Returns:
        STOI score (higher is better, range: 0 to 1).
    """
    try:
        from pystoi import stoi

        score = stoi(reference, degraded, sample_rate, extended=False)
        return float(score)
    except Exception as e:
        logger.warning(f"Error computing STOI: {e}")
        return 0.0


def compute_si_sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """Compute SI-SDR (Scale-Invariant Signal-to-Distortion Ratio).

    Args:
        reference: Reference (clean) audio.
        estimate: Estimated (enhanced) audio.

    Returns:
        SI-SDR score in dB (higher is better).
    """
    try:
        # Ensure same length
        min_len = min(len(reference), len(estimate))
        reference = reference[:min_len]
        estimate = estimate[:min_len]

        # Remove mean
        reference = reference - np.mean(reference)
        estimate = estimate - np.mean(estimate)

        # Compute SI-SDR
        reference_energy = np.sum(reference**2) + 1e-10
        optimal_scaling = np.sum(reference * estimate) / reference_energy
        projection = optimal_scaling * reference

        noise = estimate - projection
        si_sdr = 10 * np.log10(
            (np.sum(projection**2) + 1e-10) / (np.sum(noise**2) + 1e-10)
        )

        return float(si_sdr)
    except Exception as e:
        logger.warning(f"Error computing SI-SDR: {e}")
        return 0.0


def compute_phoneme_preservation(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = 16000,
) -> float:
    """Compute phoneme preservation rate.

    This metric measures how well phonetic content is preserved
    during enhancement. Higher values indicate better preservation.

    Args:
        reference: Reference (clean) audio.
        estimate: Estimated (enhanced) audio.
        sample_rate: Sample rate in Hz.

    Returns:
        Phoneme preservation rate (0 to 1).
    """
    try:
        import librosa

        # Extract MFCCs as phoneme-related features
        ref_mfccs = librosa.feature.mfcc(y=reference, sr=sample_rate, n_mfcc=13)
        est_mfccs = librosa.feature.mfcc(y=estimate, sr=sample_rate, n_mfcc=13)

        # Ensure same length
        min_frames = min(ref_mfccs.shape[1], est_mfccs.shape[1])
        ref_mfccs = ref_mfccs[:, :min_frames]
        est_mfccs = est_mfccs[:, :min_frames]

        # Compute cosine similarity for each frame
        similarities = []
        for i in range(min_frames):
            ref_vec = ref_mfccs[:, i]
            est_vec = est_mfccs[:, i]

            # Cosine similarity
            similarity = np.dot(ref_vec, est_vec) / (
                np.linalg.norm(ref_vec) * np.linalg.norm(est_vec) + 1e-10
            )
            similarities.append(similarity)

        # Average similarity
        preservation_rate = np.mean(similarities)

        # Clip to [0, 1]
        preservation_rate = np.clip(preservation_rate, 0.0, 1.0)

        return float(preservation_rate)
    except Exception as e:
        logger.warning(f"Error computing phoneme preservation: {e}")
        return 0.0


def compute_all_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = 16000,
) -> dict:
    """Compute all evaluation metrics.

    Args:
        reference: Reference (clean) audio.
        estimate: Estimated (enhanced) audio.
        sample_rate: Sample rate in Hz.

    Returns:
        Dictionary of metric scores.
    """
    metrics = {
        "pesq": compute_pesq(reference, estimate, sample_rate),
        "stoi": compute_stoi(reference, estimate, sample_rate),
        "si_sdr": compute_si_sdr(reference, estimate),
        "phoneme_preservation_rate": compute_phoneme_preservation(
            reference, estimate, sample_rate
        ),
    }

    return metrics


def batch_compute_metrics(
    references: list,
    estimates: list,
    sample_rate: int = 16000,
) -> dict:
    """Compute metrics for a batch of audio samples.

    Args:
        references: List of reference (clean) audio arrays.
        estimates: List of estimated (enhanced) audio arrays.
        sample_rate: Sample rate in Hz.

    Returns:
        Dictionary of averaged metric scores.
    """
    all_metrics = []

    for ref, est in zip(references, estimates):
        metrics = compute_all_metrics(ref, est, sample_rate)
        all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if m[key] is not None]
        avg_metrics[key] = np.mean(values) if values else 0.0
        avg_metrics[f"{key}_std"] = np.std(values) if values else 0.0

    return avg_metrics
