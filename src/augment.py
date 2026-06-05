"""Synthetic training data via template placement.

Each training window is built by placing complete alarm cycles (templates)
at random positions, then adding continuous background noise.  This
preserves the natural rhythm of each alarm (e.g. ACC's three pulses,
Caution2's on-off pattern) instead of randomly cropping which would
destroy cyclic structure.

  - noise only          (15%)  — all-negative samples
  - 1 alarm + noise     (70%)  — single isolated ring
  - 2 alarms + noise    (15%)  — overlapping scenario
"""

import numpy as np

from .config import (
    N_SAMPLES,
    MIN_SNR_DB,
    MAX_SNR_DB,
    NO_ALARM_PROB,
    SINGLE_ALARM_PROB,
    DUAL_ALARM_PROB,
    DUAL_VOLUME_RATIO,
)


def _pick_audio(pool) -> np.ndarray:
    """Return a random audio array from *pool* (single array or list)."""
    if isinstance(pool, list):
        return pool[np.random.randint(len(pool))]
    return pool


def _pick_alarms(candidates: list[int], count: int) -> list[int]:
    """Pick *count* distinct alarm indices from candidates."""
    if count > len(candidates):
        count = len(candidates)
    return [int(c) for c in np.random.choice(candidates, size=count, replace=False)]


def _snr_mix(signal: np.ndarray, noise_seg: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix signal and noise at the given SNR (dB)."""
    sig_power = np.mean(signal**2)
    noise_power = np.mean(noise_seg**2)
    if noise_power < 1e-10:
        return signal
    desired_noise_power = sig_power / (10 ** (snr_db / 10))
    scale = np.sqrt(desired_noise_power / noise_power)
    return signal + noise_seg * scale


def _get_noise_segment(noise_list: list, length: int) -> np.ndarray:
    """Return a random *length* chunk from the noise pool."""
    noise = noise_list[np.random.randint(len(noise_list))]
    if len(noise) >= length:
        start = np.random.randint(0, len(noise) - length + 1)
        return noise[start:start + length].copy()
    # Tile noise if too short
    repeats = (length // len(noise)) + 1
    return np.tile(noise, repeats)[:length]


def generate_sample(
    alarms: dict,
    noise_list: list,
    alarm_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one training sample by placing complete alarm cycles.

    Each alarm appears as its full natural cycle placed at a random offset
    within the window.  Multiple alarms may overlap.

    Returns:
        audio:  np.ndarray of shape (N_SAMPLES,), float32
        label:  np.ndarray of shape (num_classes,), float32 multi-hot
    """
    num_classes = len(alarms)
    candidates = (list(alarms.keys()) if alarm_indices is None
                  else [i for i in alarm_indices if i in alarms])

    if len(candidates) < 1:
        raise ValueError("No alarm candidates available.")

    # Decide how many alarms (0, 1, or 2)
    r = np.random.random()
    if r < NO_ALARM_PROB:
        num_alarms = 0
    elif r < NO_ALARM_PROB + SINGLE_ALARM_PROB:
        num_alarms = 1
    else:
        num_alarms = 2

    num_alarms = min(num_alarms, len(candidates))
    label = np.zeros(num_classes, dtype=np.float32)

    # --- noise-only case ---
    if num_alarms == 0:
        noise_seg = _get_noise_segment(noise_list, N_SAMPLES)
        peak = np.max(np.abs(noise_seg))
        if peak > 0:
            noise_seg = noise_seg / peak
        return noise_seg.astype(np.float32), label

    # --- Place alarm templates ---
    chosen = _pick_alarms(candidates, num_alarms)
    mixed = np.zeros(N_SAMPLES, dtype=np.float32)

    for rank, idx in enumerate(chosen):
        # Get the alarm audio (one complete cycle or longer recording)
        alarm_audio = _pick_audio(alarms[idx]).copy()
        cycle_len = len(alarm_audio)

        # If the cycle is longer than our window, take a random crop from it
        # (handles 10s recordings — still random crop, but preserves structure)
        if cycle_len > N_SAMPLES:
            start = np.random.randint(0, cycle_len - N_SAMPLES + 1)
            alarm_seg = alarm_audio[start:start + N_SAMPLES]
            offset = 0
        else:
            # Place the complete cycle at a random position in the window
            max_offset = N_SAMPLES - cycle_len
            offset = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0
            alarm_seg = alarm_audio

        # Add to mix (second alarm at variable volume for realistic overlap)
        if rank == 0:
            mixed[offset:offset + len(alarm_seg)] += alarm_seg
        else:
            vol = np.random.uniform(*DUAL_VOLUME_RATIO)
            mixed[offset:offset + len(alarm_seg)] += alarm_seg * vol

        label[idx] = 1.0

    # --- Add continuous background noise ---
    noise_seg = _get_noise_segment(noise_list, N_SAMPLES)
    snr_db = np.random.uniform(MIN_SNR_DB, MAX_SNR_DB)
    audio = _snr_mix(mixed, noise_seg, snr_db)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32), label
