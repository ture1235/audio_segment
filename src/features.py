"""Mel-spectrogram feature extraction."""

import numpy as np
import librosa

from .config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, FMAX


def audio_to_melspec(audio: np.ndarray) -> np.ndarray:
    """Convert a 1D audio array to a log-mel-spectrogram.

    Args:
        audio: float32 array of shape (samples,), normalized to [-1, 1].

    Returns:
        float32 array of shape (1, N_MELS, T) normalized to [0, 1].
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    # Normalize to [0, 1]
    mel_db = mel_db - mel_db.min()
    if mel_db.max() > 0:
        mel_db = mel_db / mel_db.max()
    return mel_db[np.newaxis, :, :].astype(np.float32)
