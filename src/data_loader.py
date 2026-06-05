"""Audio file loading and preprocessing.

Supports two directory layouts:

  Legacy flat (one file per class):
    data/alarms/ACC.wav
    data/alarms/BSW.wav

  Subdirectory (multiple files per class):
    data/alarms/ACC/cycle1.wav
    data/alarms/ACC/cycle2.wav
    data/alarms/BSW/variant_a.wav

Mixed layouts work too.  The class name is the directory name (or the
filename minus .wav for flat files).
"""

import os
import numpy as np
import soundfile as sf
import librosa

from .config import ALARM_DIR, NOISE_DIR, SAMPLE_RATE


def _load_wav(filepath: str) -> np.ndarray:
    """Load a single wav, convert to mono, resample, normalize. Returns float32."""
    audio, sr = sf.read(filepath, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def _collect_wavs(root: str) -> list[str]:
    """Return sorted list of .wav file paths under *root* (non-recursive)."""
    if not os.path.isdir(root):
        return []
    return sorted(
        os.path.join(root, f)
        for f in os.listdir(root)
        if f.lower().endswith(".wav") and not f.startswith(".")
    )


def load_alarms(alarm_dir: str = ALARM_DIR) -> tuple[dict, list[str]]:
    """Auto-discover alarm classes from directory structure.

    Two layouts are supported (can be mixed):

    1. Subdirectory layout (recommended for multiple samples per class):
         data/alarms/ACC/  → class "ACC", all .wav files inside belong to it
         data/alarms/BSW/  → class "BSW"

    2. Flat layout (single file per class):
         data/alarms/ACC.wav  → class "ACC"
         data/alarms/BSW.wav  → class "BSW"

    Returns:
        alarms:  dict[int, list[np.ndarray]]  alarm_index -> list of audio arrays
        alarm_names: list[str]                  alarm_names[i] = class name
    """
    if not os.path.isdir(alarm_dir):
        raise FileNotFoundError(f"Alarm directory not found: {alarm_dir}")

    alarms: dict[int, list[np.ndarray]] = {}
    alarm_names: list[str] = []

    # --- Pass 1: subdirectories (each dir = one class) ---
    subdirs = sorted(
        d for d in os.listdir(alarm_dir)
        if os.path.isdir(os.path.join(alarm_dir, d)) and not d.startswith(".")
    )

    for class_name in subdirs:
        class_dir = os.path.join(alarm_dir, class_name)
        wavs = _collect_wavs(class_dir)
        if not wavs:
            continue

        idx = len(alarm_names)
        alarm_names.append(class_name)
        loaded = [_load_wav(p) for p in wavs]
        alarms[idx] = loaded
        print(f"  [{idx}] {class_name:20s}  ({len(wavs)} file(s))")

    # --- Pass 2: loose .wav files (each file = one class, unless its name
    #             matches a subdirectory we already picked up) ---
    loose = [
        f for f in os.listdir(alarm_dir)
        if f.lower().endswith(".wav")
        and not f.startswith(".")
        and os.path.isfile(os.path.join(alarm_dir, f))
    ]

    for fname in sorted(loose):
        class_name = os.path.splitext(fname)[0]
        if class_name in alarm_names:
            # Already registered via a subdirectory — skip the loose file
            print(f"  [·] {fname}  (skipped, class '{class_name}' already defined by directory)")
            continue

        idx = len(alarm_names)
        alarm_names.append(class_name)
        filepath = os.path.join(alarm_dir, fname)
        alarms[idx] = [_load_wav(filepath)]
        print(f"  [{idx}] {class_name:20s}  (1 file)")

    if len(alarms) == 0:
        raise FileNotFoundError(
            f"No .wav files found in: {alarm_dir}\n"
            f"  Place alarm files directly (e.g. ACC.wav) or in subdirectories\n"
            f"  (e.g. ACC/cycle1.wav, ACC/cycle2.wav)."
        )

    return alarms, alarm_names


def _generate_synthetic_noise(
    num_clips: int = 3, duration_seconds: float = 10.0
) -> list[np.ndarray]:
    """Generate synthetic background noise as fallback when no real noise files exist.

    Produces a mix of white + pink-like noise with low-frequency emphasis,
    roughly approximating car cabin ambient sound.
    """
    n_samples = int(SAMPLE_RATE * duration_seconds)
    clips = []
    for i in range(num_clips):
        np.random.seed(i)
        white = np.random.normal(0, 0.3, n_samples).astype(np.float32)
        pink = np.convolve(white, np.ones(64) / 64, mode="same")
        t = np.arange(n_samples) / SAMPLE_RATE
        hum = 0.05 * np.sin(2 * np.pi * 120 * t) + 0.03 * np.sin(2 * np.pi * 240 * t)
        mixed = pink + hum.astype(np.float32)
        peak = np.max(np.abs(mixed))
        if peak > 0:
            mixed = mixed / peak
        clips.append(mixed.astype(np.float32))
    return clips


def load_noise(noise_dir: str = NOISE_DIR) -> list[np.ndarray]:
    """Load noise audio files, falling back to synthetic noise if none found.

    Returns list of float32 arrays, each normalized to [-1, 1].
    """
    noise_list = []

    if os.path.isdir(noise_dir):
        for fname in sorted(os.listdir(noise_dir)):
            if fname.startswith("."):
                continue
            filepath = os.path.join(noise_dir, fname)
            if not os.path.isfile(filepath):
                continue
            try:
                audio, sr = sf.read(filepath, dtype="float32")
            except Exception:
                print(f"  [WARN] Skipping unreadable file: {fname}")
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak
            noise_list.append(audio.astype(np.float32))

    if noise_list:
        return noise_list

    print("  [INFO] No noise files found, using synthetic noise fallback.")
    print("  [INFO] For better results, add real car cabin recordings to data/noise/")
    return _generate_synthetic_noise()
