"""Tests for data loading and preprocessing."""

import numpy as np
import pytest
import torch

from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.data import (
    SpeechEnhancementDataset,
    create_dataloaders,
)
from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.data.preprocessing import (
    add_noise,
    extract_mel_spectrogram,
    extract_phonemes,
    normalize_audio,
)


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_normalize_audio(self, clean_audio):
        """Test audio normalization."""
        normalized = normalize_audio(clean_audio, target_db=-20.0)

        assert normalized.shape == clean_audio.shape
        assert normalized.dtype == clean_audio.dtype
        assert not np.array_equal(normalized, clean_audio)

    def test_normalize_audio_zero(self):
        """Test normalization of zero audio."""
        zero_audio = np.zeros(1000)
        normalized = normalize_audio(zero_audio)

        assert normalized.shape == zero_audio.shape
        assert np.allclose(normalized, zero_audio)

    def test_add_noise_white(self, clean_audio):
        """Test white noise addition."""
        noisy = add_noise(clean_audio, noise_type="white", snr_db=10.0)

        assert noisy.shape == clean_audio.shape
        assert not np.array_equal(noisy, clean_audio)

        # Check SNR is approximately correct
        signal_power = np.mean(clean_audio**2)
        noise_power = np.mean((noisy - clean_audio) ** 2)
        actual_snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

        assert abs(actual_snr - 10.0) < 2.0  # Allow 2 dB tolerance

    def test_add_noise_babble(self, clean_audio):
        """Test babble noise addition."""
        noisy = add_noise(clean_audio, noise_type="babble", snr_db=15.0)

        assert noisy.shape == clean_audio.shape
        assert not np.array_equal(noisy, clean_audio)

    def test_add_noise_cafe(self, clean_audio):
        """Test cafe noise addition."""
        noisy = add_noise(clean_audio, noise_type="cafe", snr_db=5.0)

        assert noisy.shape == clean_audio.shape
        assert not np.array_equal(noisy, clean_audio)

    def test_add_noise_invalid_type(self, clean_audio):
        """Test invalid noise type."""
        with pytest.raises(ValueError):
            add_noise(clean_audio, noise_type="invalid")

    def test_extract_mel_spectrogram(self, clean_audio, sample_rate):
        """Test mel spectrogram extraction."""
        mel_spec = extract_mel_spectrogram(
            clean_audio, sample_rate=sample_rate, n_mels=128
        )

        assert isinstance(mel_spec, torch.Tensor)
        assert mel_spec.shape[0] == 128  # n_mels
        assert mel_spec.shape[1] > 0  # time dimension

    def test_extract_phonemes(self, clean_audio, sample_rate):
        """Test phoneme feature extraction."""
        phoneme_features = extract_phonemes(clean_audio, sample_rate=sample_rate)

        assert isinstance(phoneme_features, np.ndarray)
        assert phoneme_features.shape[0] == 39  # MFCC + deltas
        assert phoneme_features.shape[1] > 0  # time dimension


class TestDataset:
    """Tests for dataset class."""

    def test_dataset_creation(self, data_config):
        """Test dataset creation."""
        audio_files = ["file1.wav", "file2.wav", "file3.wav"]
        languages = ["en", "es", "cy"]

        dataset = SpeechEnhancementDataset(
            audio_files, languages, data_config, mode="train"
        )

        assert len(dataset) == 3
        assert dataset.mode == "train"

    def test_create_dataloaders(self):
        """Test dataloader creation."""
        config = {
            "seed": 42,
            "data": {
                "sample_rate": 16000,
                "n_fft": 1024,
                "hop_length": 256,
                "n_mels": 128,
                "max_audio_length": 2.0,
                "snr_range": [-5, 20],
                "noise_types": ["white"],
                "train_split": 0.7,
                "val_split": 0.2,
                "test_split": 0.1,
                "num_workers": 0,  # Use 0 for testing
            },
            "training": {"batch_size": 2},
        }

        train_loader, val_loader, test_loader = create_dataloaders(
            config, num_samples=10, use_synthetic=True
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

        # Test batch loading
        batch = next(iter(train_loader))
        assert "clean_audio" in batch
        assert "noisy_audio" in batch
        assert "clean_mel" in batch
        assert "noisy_mel" in batch
        assert "phoneme_features" in batch
        assert "language" in batch

    def test_dataloader_batch_shape(self):
        """Test dataloader batch shapes."""
        config = {
            "seed": 42,
            "data": {
                "sample_rate": 16000,
                "n_fft": 1024,
                "hop_length": 256,
                "n_mels": 128,
                "max_audio_length": 2.0,
                "snr_range": [-5, 20],
                "noise_types": ["white"],
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
                "num_workers": 0,
            },
            "training": {"batch_size": 4},
        }

        train_loader, _, _ = create_dataloaders(
            config, num_samples=20, use_synthetic=True
        )

        batch = next(iter(train_loader))

        assert batch["clean_audio"].shape[0] == 4  # batch size
        assert batch["noisy_audio"].shape[0] == 4
        assert batch["clean_mel"].shape[0] == 4
        assert batch["noisy_mel"].shape[0] == 4
        assert batch["phoneme_features"].shape[0] == 4
        assert batch["language"].shape[0] == 4
