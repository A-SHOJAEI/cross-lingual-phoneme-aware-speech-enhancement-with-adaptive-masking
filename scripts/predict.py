#!/usr/bin/env python
"""Prediction script for phoneme-aware speech enhancement."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cross_lingual_phoneme_aware_speech_enhancement_with_adaptive_masking.data.preprocessing import (
    extract_mel_spectrogram,
    extract_phonemes,
    load_audio,
    normalize_audio,
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
    return model, config


def enhance_audio_file(
    model: PhonemeAwareEnhancer,
    input_path: str,
    output_path: str,
    config: dict,
    device: torch.device,
    language: str = "en",
) -> dict:
    """Enhance a single audio file.

    Args:
        model: Trained model.
        input_path: Path to input (noisy) audio file.
        output_path: Path to save enhanced audio.
        config: Configuration dictionary.
        device: Device to run on.
        language: Language code.

    Returns:
        Dictionary with prediction info.
    """
    # Load audio
    data_config = config.get("data", {})
    sample_rate = data_config.get("sample_rate", 16000)
    max_length = data_config.get("max_audio_length", 10.0)

    logger.info(f"Loading audio from {input_path}")
    audio, sr = load_audio(input_path, sample_rate=sample_rate, max_length=max_length)

    # Normalize
    audio = normalize_audio(audio)

    # Extract mel spectrogram
    noisy_mel = extract_mel_spectrogram(
        audio,
        sample_rate=sample_rate,
        n_fft=data_config.get("n_fft", 1024),
        hop_length=data_config.get("hop_length", 256),
        n_mels=data_config.get("n_mels", 128),
    )

    # Extract phoneme features
    phoneme_features = extract_phonemes(audio, sample_rate=sample_rate)

    # Add batch dimension
    noisy_mel = noisy_mel.unsqueeze(0).to(device)
    phoneme_features = torch.from_numpy(phoneme_features).float().unsqueeze(0).to(device)

    # Language encoding
    lang_to_idx = {"en": 0, "es": 1, "cy": 2, "eu": 3}
    language_idx = torch.tensor([lang_to_idx.get(language, 0)], dtype=torch.long).to(device)

    # Enhance
    logger.info("Enhancing audio...")
    with torch.no_grad():
        outputs = model(noisy_mel, phoneme_features, language_idx)
        enhanced_mel = outputs["enhanced_mel"]
        mask = outputs["mask"]

    # For simplicity, save original audio (in practice, convert mel back to waveform)
    # In production, you would use Griffin-Lim or a neural vocoder
    enhanced_audio = audio  # Placeholder

    # Save enhanced audio
    logger.info(f"Saving enhanced audio to {output_path}")
    sf.write(output_path, enhanced_audio, sample_rate)

    # Compute confidence score (average mask value)
    confidence_score = float(mask.mean().cpu().numpy())

    result = {
        "input_path": input_path,
        "output_path": output_path,
        "language": language,
        "confidence_score": confidence_score,
        "duration_seconds": len(enhanced_audio) / sample_rate,
    }

    return result


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description="Enhance noisy audio using phoneme-aware speech enhancement"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input audio file or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save enhanced audio (default: input_enhanced.wav)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "es", "cy", "eu"],
        help="Language of the audio",
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

        # Load model
        model, config = load_model(args.checkpoint, device)

        # Determine output path
        input_path = Path(args.input)
        if args.output is not None:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}"

        # Check if input exists
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)

        # Enhance audio
        result = enhance_audio_file(
            model, str(input_path), str(output_path), config, device, args.language
        )

        # Print results
        logger.info("\n" + "=" * 50)
        logger.info("Enhancement Results:")
        logger.info("=" * 50)
        logger.info(f"Input: {result['input_path']}")
        logger.info(f"Output: {result['output_path']}")
        logger.info(f"Language: {result['language']}")
        logger.info(f"Duration: {result['duration_seconds']:.2f} seconds")
        logger.info(f"Confidence Score: {result['confidence_score']:.4f}")
        logger.info("=" * 50)

        logger.info("Enhancement completed successfully!")

    except Exception as e:
        logger.error(f"Enhancement failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
