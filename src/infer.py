"""Single-file inference for alarm sound classification with per-ring timing.

Two detection modes based on the model's probability pattern:
  - Discrete alarms (distinct peaks) → each probability peak = one ring
  - Continuous alarms (probability plateau) → rings estimated from cycle duration

Usage:
    python -m src.infer --audio recording.wav
    python -m src.infer --audio recording.wav --threshold 0.5
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import librosa
from scipy.signal import find_peaks

from .config import (
    MODEL_PATH,
    SAMPLE_RATE,
    N_SAMPLES,
    INFERENCE_THRESHOLD,
    WINDOW_STRIDE,
    ALARM_DIR,
)
from .features import audio_to_melspec
from .model import AlarmClassifier, AlarmCRNN


# ---------------------------------------------------------------------------
# Cycle-duration metadata
# ---------------------------------------------------------------------------

CYCLE_DURATIONS_PATH = os.path.join(ALARM_DIR, "cycle_durations.txt")


def load_cycle_durations(path: str = CYCLE_DURATIONS_PATH) -> dict[str, float]:
    durations: dict[str, float] = {}
    if not os.path.exists(path):
        return durations
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("=", 1)
            if len(parts) == 2:
                try:
                    durations[parts[0].strip()] = float(parts[1].strip())
                except ValueError:
                    continue
    return durations


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_path: str = MODEL_PATH) -> tuple[nn.Module, list[str]]:
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    alarm_names = checkpoint["alarm_names"]
    num_classes = len(alarm_names)
    model_type = checkpoint.get("model_type", "cnn")
    if model_type == "crnn":
        model = AlarmCRNN(num_classes=num_classes)
    else:
        model = AlarmClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, alarm_names


# ---------------------------------------------------------------------------
# Sliding-window probabilities
# ---------------------------------------------------------------------------

def sliding_window_inference(
    audio: np.ndarray,
    model: AlarmClassifier,
    window_samples: int = N_SAMPLES,
    stride: float = WINDOW_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (prob_grid, time_axis).

    prob_grid: (num_classes, num_windows)
    time_axis: (num_windows,)  — window center times in seconds
    """
    stride_samples = int(SAMPLE_RATE * stride)
    device = torch.device("cpu")

    if len(audio) < window_samples:
        audio = np.pad(audio, (0, window_samples - len(audio)))

    num_windows = max(1, (len(audio) - window_samples) // stride_samples + 1)
    time_axis = np.arange(num_windows) * stride + (window_samples / SAMPLE_RATE / 2)
    prob_grid = np.zeros((0, num_windows), dtype=np.float32)

    for i in range(num_windows):
        start_sample = i * stride_samples
        segment = audio[start_sample : start_sample + window_samples]

        peak = np.max(np.abs(segment))
        if peak > 0:
            segment = segment / peak

        melspec = audio_to_melspec(segment)
        x = torch.from_numpy(melspec).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        if i == 0:
            prob_grid = np.zeros((len(probs), num_windows), dtype=np.float32)
        prob_grid[:, i] = probs

    return prob_grid, time_axis


# ---------------------------------------------------------------------------
# Ring detection
# ---------------------------------------------------------------------------

def _is_continuous(probs: np.ndarray, threshold: float, cycle_duration: float) -> bool:
    """Return True if the longest contiguous above-threshold run spans ≥ 3 cycles.

    Uses the longest active run rather than global percentage, so a 7-second
    alarm in a 31-second recording isn't misclassified as discrete.
    """
    above = probs >= threshold
    if not above.any():
        return False

    # Find the longest contiguous run of above-threshold windows
    max_run = 0
    current_run = 0
    for a in above:
        if a:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    run_duration = max_run * WINDOW_STRIDE
    return run_duration >= cycle_duration * 3


def _detect_discrete_rings(
    probs: np.ndarray,
    times: np.ndarray,
    threshold: float,
    min_ring_spacing_ms: float,
) -> list[dict]:
    """Find individual rings as probability peaks."""
    min_spacing = max(1, int(min_ring_spacing_ms / 1000 / WINDOW_STRIDE))
    peaks, props = find_peaks(
        probs,
        height=threshold,
        distance=min_spacing,
        prominence=0.03,
    )
    rings = []
    for p_idx in peaks:
        rings.append({
            "time": float(times[p_idx]),
            "prob": float(probs[p_idx]),
        })
    return rings


def _detect_continuous_rings(
    probs: np.ndarray,
    times: np.ndarray,
    threshold: float,
    cycle_duration: float,
) -> list[dict]:
    """For a continuous alarm plateau, estimate individual ring times.

    The 2.0s sliding window smears each ring's probability, so the observed
    region is wider than reality.  Ring positions are approximated by walking
    forward from the first detection at *cycle_duration* intervals.
    Ring counts are approximate (±1–2) due to this window smearing.
    """
    above = probs >= threshold
    if not above.any():
        return []

    active_indices = np.where(above)[0]
    first_active = float(times[active_indices[0]])
    last_active = float(times[active_indices[-1]])

    # Walk forward by cycle_duration from first detection
    rings = []
    t = first_active
    while t <= last_active:
        idx = np.argmin(np.abs(times - t))
        rings.append({
            "time": float(t),
            "prob": float(probs[idx]),
        })
        t += cycle_duration

    return rings


def detect_all_rings(
    prob_grid: np.ndarray,
    time_axis: np.ndarray,
    alarm_names: list[str],
    threshold: float = INFERENCE_THRESHOLD,
    min_ring_spacing_ms: float = 300.0,
) -> dict[str, list[dict]]:
    """Detect rings for all alarm classes, handling both discrete and continuous patterns.

    Returns: class_name → list of {"time": float (s), "prob": float}
    """
    cycle_durations = load_cycle_durations()
    result: dict[str, list[dict]] = {}

    for class_idx, name in enumerate(alarm_names):
        probs = prob_grid[class_idx]
        if probs.max() < threshold:
            continue

        cycle_dur = cycle_durations.get(name)

        if cycle_dur and _is_continuous(probs, threshold, cycle_dur):
            rings = _detect_continuous_rings(probs, time_axis, threshold, cycle_dur)
        else:
            rings = _detect_discrete_rings(probs, time_axis, threshold, min_ring_spacing_ms)

        if rings:
            result[name] = rings

    return result


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_file(
    audio_path: str,
    model_path: str = MODEL_PATH,
    threshold: float = INFERENCE_THRESHOLD,
    min_ring_spacing_ms: float = 300.0,
) -> None:
    print(f"Model:    {model_path}")
    model, alarm_names = load_model(model_path)

    print(f"Audio:    {audio_path}")
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    total_dur = len(audio) / SAMPLE_RATE
    print(f"Duration: {total_dur:.1f}s  Window: {N_SAMPLES / SAMPLE_RATE:.1f}s  "
          f"Stride: {WINDOW_STRIDE}s")
    print()

    cycle_durations = load_cycle_durations()

    prob_grid, time_axis = sliding_window_inference(audio, model)
    all_rings = detect_all_rings(prob_grid, time_axis, alarm_names, threshold,
                                 min_ring_spacing_ms)

    # --- Volume gating ---
    # Compute a local noise floor: median energy across the whole audio
    # --- Output ---
    total = sum(len(v) for v in all_rings.values())
    print(f"Rings detected: {total}  |  Threshold: {threshold}  "
          f"|  Min spacing: {min_ring_spacing_ms}ms")
    print()

    if not all_rings:
        print("No alarms detected.")
        return

    for name in alarm_names:
        rings = all_rings.get(name, [])
        if not rings:
            continue

        cycle_str = ""
        if name in cycle_durations:
            cycle_str = f"  (cycle: {cycle_durations[name]*1000:.0f}ms)"

        cd = cycle_durations.get(name)
        is_continuous = cd and _is_continuous(prob_grid[alarm_names.index(name)], threshold, cd)
        mode_hint = " (approx, continuous)" if is_continuous else ""
        print(f"┌─ {name}  ×{len(rings)} rings{cycle_str}{mode_hint}")
        for i, r in enumerate(rings):
            branch = "└─" if i == len(rings) - 1 else "├─"
            mm = int(r["time"] // 60)
            ss = r["time"] % 60
            print(f"{branch} ring #{i+1}:  {mm:02d}:{ss:06.3f}  conf={r['prob']:.3f}")
        print()

    # Summary
    print("─" * 50)
    print(f"{'Class':15s}  {'Rings':>5s}  {'Time range':>20s}")
    print("─" * 50)
    for name in alarm_names:
        rings = all_rings.get(name, [])
        if not rings:
            continue
        n = len(rings)
        t_min = min(r["time"] for r in rings)
        t_max = max(r["time"] for r in rings)
        mm0, mm1 = int(t_min // 60), int(t_max // 60)
        range_str = f"{mm0:02d}:{t_min%60:06.3f} – {mm1:02d}:{t_max%60:06.3f}"
        print(f"{name:15s}  {n:>5d}  {range_str:>20s}")
    print("─" * 50)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Alarm sound classifier with per-ring timing")
    # parser.add_argument("--audio", required=True, help="Path to audio file")
    # parser.add_argument("--model", default=MODEL_PATH, help="Path to model checkpoint")
    # parser.add_argument("--threshold", type=float, default=INFERENCE_THRESHOLD,
    #                     help="Detection threshold (0–1)")
    # parser.add_argument("--min-ring-spacing", type=float, default=300.0,
    #                     help="Minimum spacing between rings (ms), for discrete mode")
    # args = parser.parse_args()
    # infer_file(args.audio, args.model, args.threshold, args.min_ring_spacing)
    wavs_folder = "./test_wav"
    for wav in os.listdir(wavs_folder):
        if wav.endswith(".wav"):
            audio_path = os.path.join(wavs_folder, wav)
            infer_file(audio_path)
            