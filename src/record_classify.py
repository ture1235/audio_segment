"""Real-time alarm classification from microphone input.

Usage:
    python -m src.record_classify
    python -m src.record_classify --threshold 0.4 --interval 0.5
"""

import argparse
import time
import threading
import queue
import numpy as np
import torch

from .config import (
    MODEL_PATH,
    SAMPLE_RATE,
    N_SAMPLES,
    INFERENCE_THRESHOLD,
    WINDOW_STRIDE,
)
from .features import audio_to_melspec
from .model import AlarmClassifier, AlarmCRNN


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


class RealtimeClassifier:
    """Continuously records from microphone and classifies alarm sounds."""

    def __init__(
        self,
        model: AlarmClassifier,
        alarm_names: list[str],
        threshold: float = INFERENCE_THRESHOLD,
        interval: float = WINDOW_STRIDE,
    ):
        self.model = model
        self.alarm_names = alarm_names
        self.threshold = threshold
        self.interval = interval  # seconds between inferences
        self.buffer = np.zeros(N_SAMPLES * 2, dtype=np.float32)  # 2s ring buffer
        self.buffer_lock = threading.Lock()
        self.running = False
        self.stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            print(f"[Audio] {status}")
        chunk = indata[:, 0] if indata.ndim > 1 else indata
        with self.buffer_lock:
            self.buffer = np.roll(self.buffer, -len(chunk))
            self.buffer[-len(chunk):] = chunk

    def _classify_loop(self):
        """Periodically run inference on the current audio buffer."""
        while self.running:
            time.sleep(self.interval)
            with self.buffer_lock:
                segment = self.buffer[-N_SAMPLES:].copy()

            peak = np.max(np.abs(segment))
            if peak < 0.001:  # silence, skip
                continue
            segment = segment / peak

            melspec = audio_to_melspec(segment)
            x = torch.from_numpy(melspec).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(x)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            detected = []
            for i, p in enumerate(probs):
                if p >= self.threshold:
                    detected.append(f"{self.alarm_names[i]}({p:.2f})")

            timestamp = time.strftime("%H:%M:%S")
            if detected:
                print(f"[{timestamp}] 告警: {', '.join(detected)}")
            else:
                print(f"[{timestamp}] (无告警)")

    def start(self):
        """Start recording and classification."""
        try:
            import sounddevice as sd
        except ImportError:
            print(
                "sounddevice not installed. Install with: pip install sounddevice\n"
                "Alternatively, use: python -m src.infer --audio <file>"
            )
            return

        self.running = True

        # Start classifier thread
        classify_thread = threading.Thread(target=self._classify_loop, daemon=True)
        classify_thread.start()

        print(f"Starting real-time classification (threshold={self.threshold})...")
        print(f"Alarm types: {', '.join(self.alarm_names)}")
        print("Press Ctrl+C to stop.\n")

        # Start microphone stream (blocks until interrupted)
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self._audio_callback,
                blocksize=int(SAMPLE_RATE * 0.1),  # 100ms chunks
            ):
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            classify_thread.join(timeout=1.0)
            print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Real-time alarm classifier")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=INFERENCE_THRESHOLD,
                        help="Detection threshold")
    parser.add_argument("--interval", type=float, default=WINDOW_STRIDE,
                        help="Seconds between inference calls")
    args = parser.parse_args()

    model, alarm_names = load_model(args.model)
    classifier = RealtimeClassifier(
        model, alarm_names,
        threshold=args.threshold,
        interval=args.interval,
    )
    classifier.start()


if __name__ == "__main__":
    main()
