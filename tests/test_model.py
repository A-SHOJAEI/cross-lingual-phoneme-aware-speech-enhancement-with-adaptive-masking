"""Tests for model architecture."""

import pytest
import torch

from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.models import (
    PhonemeAwareEnhancer,
)
from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.models.components import (
    AdaptiveMaskingLayer,
    ContrastivePhonemeLoss,
    PerceptualLoss,
    PhonemeAttention,
    PhonemePreservationLoss,
)


class TestComponents:
    """Tests for model components."""

    def test_phoneme_attention(self):
        """Test phoneme attention module."""
        feature_dim = 64
        phoneme_dim = 32
        batch_size = 2
        time_steps = 50

        attention = PhonemeAttention(
            feature_dim=feature_dim,
            phoneme_dim=phoneme_dim,
            num_heads=4,
        )

        audio_features = torch.randn(batch_size, time_steps, feature_dim)
        phoneme_features = torch.randn(batch_size, time_steps, phoneme_dim)

        output = attention(audio_features, phoneme_features)

        assert output.shape == (batch_size, time_steps, feature_dim)

    def test_contrastive_phoneme_loss(self):
        """Test contrastive phoneme loss."""
        loss_fn = ContrastivePhonemeLoss(temperature=0.07)

        batch_size = 4
        dim = 64

        features_1 = torch.randn(batch_size, dim)
        features_2 = torch.randn(batch_size, dim)
        positive_pairs = torch.eye(batch_size)

        loss = loss_fn(features_1, features_2, positive_pairs)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0

    def test_perceptual_loss(self, clean_audio, sample_rate):
        """Test perceptual loss."""
        loss_fn = PerceptualLoss(sample_rate=sample_rate)

        enhanced = torch.from_numpy(clean_audio).unsqueeze(0)
        clean = torch.from_numpy(clean_audio).unsqueeze(0)

        loss = loss_fn(enhanced, clean)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_phoneme_preservation_loss(self, clean_audio):
        """Test phoneme preservation loss."""
        loss_fn = PhonemePreservationLoss()

        enhanced = torch.from_numpy(clean_audio).unsqueeze(0)
        clean = torch.from_numpy(clean_audio).unsqueeze(0)

        loss = loss_fn(enhanced, clean)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_adaptive_masking_layer(self):
        """Test adaptive masking layer."""
        feature_dim = 64
        num_freq_bins = 128
        batch_size = 2
        time_steps = 50

        masking = AdaptiveMaskingLayer(
            feature_dim=feature_dim,
            num_freq_bins=num_freq_bins,
        )

        features = torch.randn(batch_size, time_steps, feature_dim)
        spectrogram = torch.randn(batch_size, num_freq_bins, time_steps)

        masked_spec, mask = masking(features, spectrogram)

        assert masked_spec.shape == spectrogram.shape
        assert mask.shape == (batch_size, num_freq_bins)
        assert torch.all(mask >= 0) and torch.all(mask <= 1)


class TestModel:
    """Tests for main model."""

    def test_model_creation(self, model_config):
        """Test model creation."""
        model = PhonemeAwareEnhancer(**model_config)

        assert isinstance(model, PhonemeAwareEnhancer)
        assert model.count_parameters() > 0

    def test_model_forward(
        self,
        model_config,
        sample_mel_spectrogram,
        sample_phoneme_features,
        sample_language,
    ):
        """Test model forward pass."""
        model = PhonemeAwareEnhancer(**model_config)

        outputs = model(
            sample_mel_spectrogram,
            sample_phoneme_features,
            sample_language,
        )

        assert "enhanced_mel" in outputs
        assert "mask" in outputs
        assert "phoneme_embeddings" in outputs

        assert outputs["enhanced_mel"].shape == sample_mel_spectrogram.shape

    def test_model_forward_without_phonemes(self, model_config, sample_mel_spectrogram):
        """Test model forward without phoneme features."""
        model = PhonemeAwareEnhancer(**model_config)

        outputs = model(sample_mel_spectrogram)

        assert "enhanced_mel" in outputs
        assert outputs["enhanced_mel"].shape == sample_mel_spectrogram.shape

    def test_model_enhance_audio(
        self,
        model_config,
        sample_mel_spectrogram,
        sample_phoneme_features,
        sample_language,
    ):
        """Test model enhancement in inference mode."""
        model = PhonemeAwareEnhancer(**model_config)

        enhanced_mel = model.enhance_audio(
            sample_mel_spectrogram,
            sample_phoneme_features,
            sample_language,
        )

        assert enhanced_mel.shape == sample_mel_spectrogram.shape

    def test_model_ablation_no_contrastive(
        self,
        sample_mel_spectrogram,
        sample_phoneme_features,
        sample_language,
    ):
        """Test model without contrastive alignment (ablation)."""
        model = PhonemeAwareEnhancer(
            n_mels=128,
            encoder_dim=64,
            decoder_dim=64,
            phoneme_embedding_dim=32,
            num_phonemes=128,
            num_attention_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout=0.1,
            use_contrastive_alignment=False,  # Ablation
            cross_lingual_transfer=False,  # Ablation
        )

        outputs = model(
            sample_mel_spectrogram,
            sample_phoneme_features,
            sample_language,
        )

        assert "enhanced_mel" in outputs
        assert outputs["enhanced_mel"].shape == sample_mel_spectrogram.shape

    def test_model_gradient_flow(
        self,
        model_config,
        sample_mel_spectrogram,
        sample_phoneme_features,
        sample_language,
    ):
        """Test gradient flow through model."""
        model = PhonemeAwareEnhancer(**model_config)

        outputs = model(
            sample_mel_spectrogram,
            sample_phoneme_features,
            sample_language,
        )

        # Compute dummy loss
        loss = outputs["enhanced_mel"].mean()
        loss.backward()

        # Check that gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break

        assert has_gradients

    def test_model_device_transfer(self, model_config):
        """Test model device transfer."""
        model = PhonemeAwareEnhancer(**model_config)

        # Test CPU
        model = model.to("cpu")
        assert next(model.parameters()).device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model = model.to("cuda")
            assert next(model.parameters()).device.type == "cuda"
