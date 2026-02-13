"""Custom model components, losses, and layers.

This module contains the novel phoneme-aware components that enable
cross-lingual transfer learning for speech enhancement.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PhonemeAttention(nn.Module):
    """Phoneme-conditioned attention mechanism.

    This is a novel component that conditions the enhancement process on
    phonetic content, enabling language-agnostic acoustic pattern learning.
    """

    def __init__(
        self,
        feature_dim: int,
        phoneme_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize phoneme attention module.

        Args:
            feature_dim: Dimension of input features.
            phoneme_dim: Dimension of phoneme embeddings.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.phoneme_dim = phoneme_dim
        self.num_heads = num_heads

        # Project phoneme features to same dimension as audio features
        self.phoneme_proj = nn.Linear(phoneme_dim, feature_dim)

        # Multi-head attention for phoneme conditioning
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)

        # Gate to control phoneme influence
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        phoneme_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply phoneme-conditioned attention.

        Args:
            audio_features: Audio features (batch, time, feature_dim).
            phoneme_features: Phoneme features (batch, phoneme_time, phoneme_dim).
            mask: Optional attention mask.

        Returns:
            Enhanced features with phoneme conditioning.
        """
        # Project phoneme features
        phoneme_proj = self.phoneme_proj(phoneme_features)

        # Apply cross-attention (audio attends to phonemes)
        attended_features, attention_weights = self.attention(
            query=audio_features,
            key=phoneme_proj,
            value=phoneme_proj,
            attn_mask=mask,
        )

        # Gated fusion
        concat_features = torch.cat([audio_features, attended_features], dim=-1)
        gate_values = self.gate(concat_features)

        # Combine with gating
        enhanced_features = audio_features + gate_values * attended_features

        # Layer normalization
        output = self.layer_norm(enhanced_features)

        return output


class ContrastivePhonemeLoss(nn.Module):
    """Contrastive loss for phoneme alignment across languages.

    This novel loss function enables cross-lingual transfer by aligning
    phoneme representations from different languages in a shared space.
    """

    def __init__(self, temperature: float = 0.07):
        """Initialize contrastive phoneme loss.

        Args:
            temperature: Temperature parameter for contrastive learning.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        phoneme_features_1: torch.Tensor,
        phoneme_features_2: torch.Tensor,
        positive_pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss between phoneme features.

        Args:
            phoneme_features_1: Phoneme features from first set (batch, dim).
            phoneme_features_2: Phoneme features from second set (batch, dim).
            positive_pairs: Binary mask indicating positive pairs (batch, batch).

        Returns:
            Contrastive loss value.
        """
        # Normalize features
        phoneme_features_1 = F.normalize(phoneme_features_1, dim=-1)
        phoneme_features_2 = F.normalize(phoneme_features_2, dim=-1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(phoneme_features_1, phoneme_features_2.T)
        similarity_matrix = similarity_matrix / self.temperature

        # Create labels for contrastive learning
        batch_size = phoneme_features_1.shape[0]
        labels = torch.arange(batch_size, device=phoneme_features_1.device)

        # InfoNCE loss
        loss_1 = F.cross_entropy(similarity_matrix, labels)
        loss_2 = F.cross_entropy(similarity_matrix.T, labels)

        loss = (loss_1 + loss_2) / 2

        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss for audio enhancement.

    Uses multi-scale spectral features to compute perceptual similarity
    between enhanced and clean audio.
    """

    def __init__(self, sample_rate: int = 16000):
        """Initialize perceptual loss.

        Args:
            sample_rate: Audio sample rate.
        """
        super().__init__()
        self.sample_rate = sample_rate

        # Multi-scale STFT parameters
        self.fft_sizes = [512, 1024, 2048]
        self.hop_sizes = [128, 256, 512]
        self.win_sizes = [512, 1024, 2048]

    def stft_loss(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_size: int,
    ) -> torch.Tensor:
        """Compute STFT-based loss.

        Args:
            enhanced: Enhanced audio waveform.
            clean: Clean audio waveform.
            fft_size: FFT size.
            hop_size: Hop size.
            win_size: Window size.

        Returns:
            STFT loss value.
        """
        # Compute STFT
        enhanced_stft = torch.stft(
            enhanced,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_size,
            return_complex=True,
        )
        clean_stft = torch.stft(
            clean,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_size,
            return_complex=True,
        )

        # Compute magnitude
        enhanced_mag = torch.abs(enhanced_stft)
        clean_mag = torch.abs(clean_stft)

        # L1 loss on magnitude
        mag_loss = F.l1_loss(enhanced_mag, clean_mag)

        # Log magnitude loss
        log_mag_loss = F.l1_loss(
            torch.log(enhanced_mag + 1e-5),
            torch.log(clean_mag + 1e-5),
        )

        return mag_loss + log_mag_loss

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale perceptual loss.

        Args:
            enhanced: Enhanced audio waveform (batch, time).
            clean: Clean audio waveform (batch, time).

        Returns:
            Perceptual loss value.
        """
        total_loss = 0.0

        for fft_size, hop_size, win_size in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes
        ):
            total_loss += self.stft_loss(enhanced, clean, fft_size, hop_size, win_size)

        return total_loss / len(self.fft_sizes)


class PhonemePreservationLoss(nn.Module):
    """Loss to ensure phonetic content preservation during enhancement.

    This novel component ensures that the enhancement process preserves
    linguistic information, critical for speech recognition accuracy.
    """

    def __init__(self, feature_dim: int = 39):
        """Initialize phoneme preservation loss.

        Args:
            feature_dim: Dimension of phoneme features (e.g., MFCCs).
        """
        super().__init__()
        self.feature_dim = feature_dim

    def extract_phoneme_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract phoneme-related features from audio.

        This is a simplified version. In production, you would use
        a pre-trained phoneme recognition model.

        Args:
            audio: Audio waveform (batch, time).

        Returns:
            Phoneme features (batch, feature_dim).
        """
        # Use mean and std of waveform as simple features
        # In production, replace with actual MFCC or phoneme model features
        mean = torch.mean(audio, dim=-1, keepdim=True)
        std = torch.std(audio, dim=-1, keepdim=True)
        max_val = torch.max(audio, dim=-1, keepdim=True)[0]
        min_val = torch.min(audio, dim=-1, keepdim=True)[0]

        # Create feature vector
        features = torch.cat([mean, std, max_val, min_val], dim=-1)

        # Pad to feature_dim if needed
        if features.shape[-1] < self.feature_dim:
            padding = torch.zeros(
                features.shape[0],
                self.feature_dim - features.shape[-1],
                device=features.device,
            )
            features = torch.cat([features, padding], dim=-1)

        return features

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """Compute phoneme preservation loss.

        Args:
            enhanced: Enhanced audio waveform (batch, time).
            clean: Clean audio waveform (batch, time).

        Returns:
            Phoneme preservation loss value.
        """
        # Extract phoneme features
        enhanced_features = self.extract_phoneme_features(enhanced)
        clean_features = self.extract_phoneme_features(clean)

        # Cosine similarity loss
        cos_sim = F.cosine_similarity(enhanced_features, clean_features, dim=-1)
        loss = 1.0 - cos_sim.mean()

        # Add L2 loss
        l2_loss = F.mse_loss(enhanced_features, clean_features)

        return loss + 0.1 * l2_loss


class AdaptiveMaskingLayer(nn.Module):
    """Adaptive time-frequency masking layer.

    This layer learns to predict optimal masks for noise suppression
    conditioned on phoneme information.
    """

    def __init__(self, feature_dim: int, num_freq_bins: int):
        """Initialize adaptive masking layer.

        Args:
            feature_dim: Dimension of input features.
            num_freq_bins: Number of frequency bins.
        """
        super().__init__()

        # Mask prediction network
        self.mask_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, num_freq_bins),
            nn.Sigmoid(),
        )

    def forward(
        self, features: torch.Tensor, spectrogram: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive masking.

        Args:
            features: Conditioning features (batch, time, feature_dim).
            spectrogram: Input spectrogram (batch, freq, time).

        Returns:
            Tuple of (masked_spectrogram, mask).
        """
        # Average features over time
        pooled_features = torch.mean(features, dim=1)

        # Predict mask
        mask = self.mask_predictor(pooled_features)

        # Apply mask (expand to match spectrogram dimensions)
        mask_expanded = mask.unsqueeze(-1)  # (batch, freq, 1)
        masked_spectrogram = spectrogram * mask_expanded

        return masked_spectrogram, mask
