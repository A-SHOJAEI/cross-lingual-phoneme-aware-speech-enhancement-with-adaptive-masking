"""Main model architecture for phoneme-aware speech enhancement."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .components import AdaptiveMaskingLayer, PhonemeAttention

logger = logging.getLogger(__name__)


class PhonemeAwareEnhancer(nn.Module):
    """Phoneme-aware speech enhancement model.

    This model uses cross-lingual phoneme embeddings to guide adaptive
    time-frequency masking for noise reduction in low-resource languages.
    """

    def __init__(
        self,
        n_mels: int = 128,
        encoder_dim: int = 256,
        decoder_dim: int = 256,
        phoneme_embedding_dim: int = 128,
        num_phonemes: int = 128,
        num_attention_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        use_contrastive_alignment: bool = True,
        cross_lingual_transfer: bool = True,
    ):
        """Initialize phoneme-aware enhancer.

        Args:
            n_mels: Number of mel frequency bins.
            encoder_dim: Encoder hidden dimension.
            decoder_dim: Decoder hidden dimension.
            phoneme_embedding_dim: Phoneme embedding dimension.
            num_phonemes: Number of phoneme classes.
            num_attention_heads: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            dropout: Dropout rate.
            use_contrastive_alignment: Whether to use contrastive phoneme alignment.
            cross_lingual_transfer: Whether to enable cross-lingual transfer.
        """
        super().__init__()

        self.n_mels = n_mels
        self.encoder_dim = encoder_dim
        self.use_contrastive_alignment = use_contrastive_alignment
        self.cross_lingual_transfer = cross_lingual_transfer

        # Input projection
        self.input_proj = nn.Linear(n_mels, encoder_dim)

        # Phoneme embedding
        self.phoneme_embedding = nn.Embedding(num_phonemes, phoneme_embedding_dim)

        # Phoneme feature projection
        self.phoneme_proj = nn.Linear(39, phoneme_embedding_dim)  # 39 = MFCC dim

        # Encoder with phoneme attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=num_attention_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Phoneme attention module (novel component)
        if self.use_contrastive_alignment or self.cross_lingual_transfer:
            self.phoneme_attention = PhonemeAttention(
                feature_dim=encoder_dim,
                phoneme_dim=phoneme_embedding_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
            )

        # Adaptive masking layer (novel component)
        self.adaptive_masking = AdaptiveMaskingLayer(
            feature_dim=encoder_dim,
            num_freq_bins=n_mels,
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_attention_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(decoder_dim, n_mels)

        # Language embedding for cross-lingual transfer
        if self.cross_lingual_transfer:
            self.language_embedding = nn.Embedding(4, encoder_dim)  # 4 languages

    def encode_phonemes(
        self, phoneme_features: torch.Tensor
    ) -> torch.Tensor:
        """Encode phoneme features.

        Args:
            phoneme_features: Phoneme features (batch, phoneme_dim, time).

        Returns:
            Phoneme embeddings (batch, time, embedding_dim).
        """
        # Transpose to (batch, time, phoneme_dim)
        phoneme_features = phoneme_features.transpose(1, 2)

        # Project to embedding space
        phoneme_embeddings = self.phoneme_proj(phoneme_features)

        return phoneme_embeddings

    def forward(
        self,
        noisy_mel: torch.Tensor,
        phoneme_features: Optional[torch.Tensor] = None,
        language: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            noisy_mel: Noisy mel spectrogram (batch, n_mels, time).
            phoneme_features: Phoneme features (batch, phoneme_dim, time).
            language: Language indices (batch,).

        Returns:
            Dictionary containing:
                - enhanced_mel: Enhanced mel spectrogram
                - mask: Predicted time-frequency mask
                - phoneme_embeddings: Phoneme embeddings (if applicable)
        """
        batch_size, n_mels, time_steps = noisy_mel.shape

        # Transpose to (batch, time, n_mels)
        noisy_mel_t = noisy_mel.transpose(1, 2)

        # Input projection
        encoder_input = self.input_proj(noisy_mel_t)

        # Add language embedding if cross-lingual transfer is enabled
        if self.cross_lingual_transfer and language is not None:
            lang_emb = self.language_embedding(language).unsqueeze(1)
            encoder_input = encoder_input + lang_emb

        # Encode
        encoder_output = self.encoder(encoder_input)

        # Apply phoneme attention if enabled
        if (self.use_contrastive_alignment or self.cross_lingual_transfer) and phoneme_features is not None:
            phoneme_embeddings = self.encode_phonemes(phoneme_features)
            encoder_output = self.phoneme_attention(
                encoder_output,
                phoneme_embeddings,
            )
        else:
            phoneme_embeddings = None

        # Apply adaptive masking
        masked_mel, mask = self.adaptive_masking(encoder_output, noisy_mel)

        # Prepare decoder input (use masked spectrogram)
        masked_mel_t = masked_mel.transpose(1, 2)
        decoder_input = self.input_proj(masked_mel_t)

        # Decode
        decoder_output = self.decoder(decoder_input, encoder_output)

        # Output projection
        enhanced_mel = self.output_proj(decoder_output)

        # Transpose back to (batch, n_mels, time)
        enhanced_mel = enhanced_mel.transpose(1, 2)

        return {
            "enhanced_mel": enhanced_mel,
            "mask": mask,
            "phoneme_embeddings": phoneme_embeddings,
            "encoder_output": encoder_output,
        }

    def enhance_audio(
        self,
        noisy_mel: torch.Tensor,
        phoneme_features: Optional[torch.Tensor] = None,
        language: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Enhance audio (inference mode).

        Args:
            noisy_mel: Noisy mel spectrogram (batch, n_mels, time).
            phoneme_features: Phoneme features (batch, phoneme_dim, time).
            language: Language indices (batch,).

        Returns:
            Enhanced mel spectrogram.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(noisy_mel, phoneme_features, language)
        return outputs["enhanced_mel"]

    def count_parameters(self) -> int:
        """Count trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
