"""Training script for the multi-label alarm classifier."""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from tqdm import tqdm

from .config import (
    MODEL_PATH,
    MODEL_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    TRAIN_SAMPLES_PER_EPOCH,
    VAL_SAMPLES,
    MODEL_TYPE,
    FINETUNE_FROM,
    FINETUNE_LR,
    FREEZE_BACKBONE,
    SAMPLE_RATE,
)
from .data_loader import load_alarms, load_noise
from .dataset import AlarmDataset
from .model import AlarmClassifier, AlarmCRNN
from .features import audio_to_melspec
from .config import ALARM_DIR, SAMPLE_RATE, N_SAMPLES, WINDOW_STRIDE


# ---------------------------------------------------------------------------
# Real validation data
# ---------------------------------------------------------------------------

def _load_real_val_data(alarm_names: list[str]) -> list[dict]:
    """Load real validation files and their expected labels.

    Reads data/real_val/labels.txt.  Each line:  filename.wav=CLASS1,CLASS2,...

    Returns list of dicts:
      {"path": str, "audio": np.ndarray, "label": np.ndarray (multi-hot)}
    """
    import librosa
    import soundfile as sf

    real_val_dir = os.path.join(os.path.dirname(ALARM_DIR), "real_val")
    label_path = os.path.join(real_val_dir, "labels.txt")
    if not os.path.exists(label_path):
        return []

    samples: list[dict] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("=", 1)
            if len(parts) != 2:
                continue
            filename = parts[0].strip()
            class_names = [c.strip() for c in parts[1].split(",") if c.strip()]

            filepath = os.path.join(real_val_dir, filename)
            if not os.path.exists(filepath):
                print(f"  [WARN] Real-val file not found: {filepath}")
                continue

            audio, sr = sf.read(filepath, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak
            audio = audio.astype(np.float32)

            label = np.zeros(len(alarm_names), dtype=np.float32)
            for cn in class_names:
                if cn in alarm_names:
                    label[alarm_names.index(cn)] = 1.0

            samples.append({"path": filepath, "audio": audio, "label": label})

    return samples


def _evaluate_real(
    model: nn.Module,
    real_samples: list[dict],
    alarm_names: list[str],
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model on real validation recordings.

    Runs sliding-window inference on each file and checks which expected
    alarms are detected in at least one window.

    Returns dict with per-class detection stats.
    """
    if not real_samples:
        return {}

    model.eval()
    stride_samples = int(SAMPLE_RATE * WINDOW_STRIDE)

    # Per-class: expected count, detected count
    per_class_expected = np.zeros(len(alarm_names), dtype=int)
    per_class_detected = np.zeros(len(alarm_names), dtype=int)
    file_accuracy = 0

    for sample in real_samples:
        audio = sample["audio"]
        expected = sample["label"]
        expected_classes = set(np.where(expected > 0)[0])

        for c in expected_classes:
            per_class_expected[c] += 1

        if len(audio) < N_SAMPLES:
            audio = np.pad(audio, (0, N_SAMPLES - len(audio)))

        num_windows = max(1, (len(audio) - N_SAMPLES) // stride_samples + 1)
        detected_max = np.zeros(len(alarm_names), dtype=np.float32)

        for i in range(num_windows):
            start = i * stride_samples
            segment = audio[start:start + N_SAMPLES]
            peak = np.max(np.abs(segment))
            if peak > 0:
                segment = segment / peak
            melspec = audio_to_melspec(segment)
            x = torch.from_numpy(melspec).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            detected_max = np.maximum(detected_max, probs)

        detected_classes = set(np.where(detected_max >= threshold)[0])

        for c in expected_classes:
            if c in detected_classes:
                per_class_detected[c] += 1

        if expected_classes == detected_classes:
            file_accuracy += 1

    return {
        "per_class_expected": per_class_expected,
        "per_class_detected": per_class_detected,
        "file_accuracy": file_accuracy,
        "total_files": len(real_samples),
    }


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Compute per-class and overall metrics on a validation set."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).int().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(batch_y.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train(alarm_dir: str | None = None, noise_dir: str | None = None):
    """Run the full training pipeline.

    All alarm types appear in both training and validation (sample-level
    split via on-the-fly synthetic generation).  This is the correct
    approach when you have few base samples per class.
    """
    # --- Load data ---
    print("[1/3] Loading alarm and noise files...")
    alarms, alarm_names = load_alarms()
    noise_list = load_noise()

    num_alarms = len(alarms)
    if num_alarms == 0:
        print("ERROR: No alarm files loaded. Check data/alarms/")
        sys.exit(1)
    print(f"  Loaded {num_alarms} alarm types, {len(noise_list)} noise file(s)")
    for i, name in enumerate(alarm_names):
        duration = len(alarms[i]) / 22050
        print(f"    [{i}] {name:20s} ({duration:.1f}s)")

    # --- DataLoaders ---
    # Both train and val use all alarm types — synthetic generation
    # ensures different samples each epoch.
    print(f"\n[2/3] Building dataloaders "
          f"(train={TRAIN_SAMPLES_PER_EPOCH}, val={VAL_SAMPLES})...")
    train_ds = AlarmDataset(
        alarms, noise_list, num_alarms,
        alarm_indices=None,
        length=TRAIN_SAMPLES_PER_EPOCH,
    )
    val_ds = AlarmDataset(
        alarms, noise_list, num_alarms,
        alarm_indices=None,
        length=VAL_SAMPLES,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    # --- Model ---
    device = torch.device("cpu")
    print(f"\n[3/3] Building model (device: {device}, type: {MODEL_TYPE})...")
    if MODEL_TYPE == "crnn":
        model = AlarmCRNN(num_classes=num_alarms).to(device)
    else:
        model = AlarmClassifier(num_classes=num_alarms).to(device)

    # Fine-tuning: load previous checkpoint, replace classifier head
    lr = LEARNING_RATE
    if FINETUNE_FROM and os.path.exists(FINETUNE_FROM):
        print(f"  Fine-tuning from: {FINETUNE_FROM}")
        ckpt = torch.load(FINETUNE_FROM, map_location=device, weights_only=False)
        old_names = ckpt.get("alarm_names", [])
        old_state = ckpt["model_state_dict"]
        new_state = model.state_dict()

        # Copy matching backbone weights (everything except classifier head)
        matched = 0
        for k, v in old_state.items():
            if k in new_state and v.shape == new_state[k].shape:
                new_state[k] = v
                matched += 1
        model.load_state_dict(new_state)
        print(f"  Copied {matched}/{len(new_state)} param tensors from previous model")
        print(f"  Old classes: {old_names} → New classes: {alarm_names}")

        if FREEZE_BACKBONE:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
            print("  Backbone frozen (classifier only)")

        lr = FINETUNE_LR
        print(f"  LR: {lr}")

    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  (trainable: {trainable:,})")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    # Per-class loss weights: give Caution1 a slight boost to prevent
    # it from being suppressed by aggressive negative sampling.
    class_weights = torch.ones(num_alarms, device=device)
    for i, name in enumerate(alarm_names):
        if name == "Caution1":
            class_weights[i] = 1.5
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # --- Model save path with timestamp ---
    run_dir = os.path.join(MODEL_DIR, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "best_model.pt")
    print(f"  Saving to: {run_dir}/")

    # --- Load real validation data ---
    real_val_samples = _load_real_val_data(alarm_names)
    if real_val_samples:
        print(f"  Real val files: {len(real_val_samples)}")
        for s in real_val_samples:
            expected = [alarm_names[i] for i, v in enumerate(s["label"]) if v > 0]
            fname = os.path.basename(s["path"])
            print(f"    {fname}: expected={expected}")

    # --- Training loop ---
    print(f"\nTraining ({EPOCHS} epochs max, patience={EARLY_STOPPING_PATIENCE})...")
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{EPOCHS}", unit="batch")
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        # Validation
        metrics = _evaluate(model, val_loader, device)
        val_f1 = metrics["macro_f1"]

        print(
            f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | "
            f"val_macro_f1={val_f1:.4f} | val_micro_f1={metrics['micro_f1']:.4f}"
        )

        # Per-class F1
        per_class = metrics["per_class_f1"]
        for i in range(len(per_class)):
            print(f"    [{i}] {alarm_names[i]:20s} F1={per_class[i]:.3f}")

        # Real validation
        if real_val_samples:
            real_metrics = _evaluate_real(
                model, real_val_samples, alarm_names, device, threshold=0.5
            )
            if real_metrics:
                print(f"  Real-val: {real_metrics['file_accuracy']}/{real_metrics['total_files']} files OK")
                for i, name in enumerate(alarm_names):
                    exp = real_metrics["per_class_expected"][i]
                    det = real_metrics["per_class_detected"][i]
                    if exp > 0:
                        status = "✓" if det == exp else f"{det}/{exp}"
                        print(f"    [{i}] {name:20s} {status}")
                print()

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "alarm_names": alarm_names,
                    "val_f1": val_f1,
                    "model_type": MODEL_TYPE,
                },
                model_path,
            )
            print(f"  -> Best model saved (macro_f1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best epoch: {best_epoch}, best macro_f1: {best_val_f1:.4f}")
    print(f"Model saved to: {model_path}")

    # Also symlink as the default for inference convenience
    legacy_path = os.path.join(MODEL_DIR, "best_model.pt")
    if os.path.islink(legacy_path) or os.path.exists(legacy_path):
        os.remove(legacy_path)
    try:
        os.symlink(model_path, legacy_path)
        print(f"Default model → {legacy_path}")
    except OSError:
        pass

    # Final evaluation report
    print("\n--- Classification Report (threshold=0.5) ---")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = _evaluate(model, val_loader, device)
    print(
        classification_report(
            final_metrics["y_true"],
            final_metrics["y_pred"],
            target_names=alarm_names,
            zero_division=0,
        )
    )
    return model, alarm_names


if __name__ == "__main__":
    train()
