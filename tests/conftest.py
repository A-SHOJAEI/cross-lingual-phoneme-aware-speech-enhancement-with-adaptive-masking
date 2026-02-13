"""Pytest configuration and fixtures."""

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_rate():
    """Sample rate fixture."""
    return 16000


@pytest.fixture
def audio_duration():
    """Audio duration fixture."""
    return 2.0


@pytest.fixture
def clean_audio(sample_rate, audio_duration):
    """Generate clean audio fixture."""
    t = np.linspace(0, audio_duration, int(sample_rate * audio_duration))
    # Simple sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


@pytest.fixture
def noisy_audio(clean_audio):
    """Generate noisy audio fixture."""
    noise = 0.1 * np.random.randn(len(clean_audio))
    return (clean_audio + noise).astype(np.float32)


@pytest.fixture
def batch_size():
    """Batch size fixture."""
    return 4


@pytest.fixture
def n_mels():
    """Number of mel bands fixture."""
    return 128


@pytest.fixture
def time_steps():
    """Number of time steps fixture."""
    return 100


@pytest.fixture
def phoneme_dim():
    """Phoneme feature dimension fixture."""
    return 39


@pytest.fixture
def sample_mel_spectrogram(batch_size, n_mels, time_steps):
    """Generate sample mel spectrogram fixture."""
    return torch.randn(batch_size, n_mels, time_steps)


@pytest.fixture
def sample_phoneme_features(batch_size, phoneme_dim, time_steps):
    """Generate sample phoneme features fixture."""
    return torch.randn(batch_size, phoneme_dim, time_steps)


@pytest.fixture
def sample_language(batch_size):
    """Generate sample language indices fixture."""
    return torch.randint(0, 4, (batch_size,))


@pytest.fixture
def model_config():
    """Model configuration fixture."""
    return {
        "n_mels": 128,
        "encoder_dim": 64,  # Smaller for testing
        "decoder_dim": 64,
        "phoneme_embedding_dim": 32,
        "num_phonemes": 128,
        "num_attention_heads": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dropout": 0.1,
        "use_contrastive_alignment": True,
        "cross_lingual_transfer": True,
    }


@pytest.fixture
def data_config():
    """Data configuration fixture."""
    return {
        "sample_rate": 16000,
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mels": 128,
        "max_audio_length": 5.0,
        "snr_range": [-5, 20],
        "noise_types": ["white", "babble", "cafe"],
    }
