"""Global configuration for the alarm sound classification system.

Training uses clean single-cycle alarm WAVs as base material and generates
synthetic multi-alarm + noise mixtures on the fly.  Inference handles
overlapping alarms (multi-label).
"""

import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ALARM_DIR = os.path.join(DATA_DIR, "alarms")
NOISE_DIR = os.path.join(DATA_DIR, "noise")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
# MODEL_PATH is set dynamically in train.py with a timestamp subdirectory.
# Legacy fallback for inference when no specific run is specified:
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")

# --- Audio Processing ---
SAMPLE_RATE = 22050
DURATION = 1.0          # seconds: fits Caution2 (1000ms) + 200ms context buffer.
# 1.2s captures the full cycle of every alarm, gives the GRU enough
# temporal context to learn on-off patterns, and is short enough that
# sliding windows (0.1s stride) cleanly separate consecutive events.
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# --- Mel-Spectrogram ---
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMAX = SAMPLE_RATE // 2

# --- Augmentation ---
MIN_SNR_DB = -5         # minimum SNR for noise mixing (dB)
MAX_SNR_DB = 20         # maximum SNR for noise mixing (dB)
NO_ALARM_PROB = 0.20      # probability of noise-only (all-negative) sample
SINGLE_ALARM_PROB = 0.60  # probability of 1 alarm + noise
DUAL_ALARM_PROB = 0.20    # probability of 2 alarms mixed + noise
TRIPLE_ALARM_PROB = 0.0   # no 3-alarm mixing
DUAL_VOLUME_RATIO = (0.3, 1.0)  # relative volume range for additional alarms

# --- Fine-tuning ---
# Set to a previous model checkpoint to continue training instead of starting
# fresh.  Useful when adding new alarm classes — backbone weights are reused,
# only the classifier head is replaced to match the new class count.
FINETUNE_FROM = None       # e.g. "models/run_20260527_114632/best_model.pt"
FINETUNE_LR = 3e-4         # lower LR for fine-tuning
FREEZE_BACKBONE = False    # freeze CNN+GRU, only train new classifier head

# --- Model ---
# NUM_CLASSES is auto-detected from alarm files at runtime
MODEL_TYPE = "crnn"       # "cnn" = CNN only,  "crnn" = CNN + GRU

# --- Training ---
BATCH_SIZE = 64          # samples per gradient step
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30              # max epochs (early stopping will likely stop sooner)
EARLY_STOPPING_PATIENCE = 12
TRAIN_SAMPLES_PER_EPOCH = 4000  # samples per epoch
VAL_SAMPLES = 1000

# --- Inference ---
INFERENCE_THRESHOLD = 0.6
WINDOW_STRIDE = 0.10    # seconds between consecutive inference windows
