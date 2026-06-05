"""CNN (baseline) and CNN+GRU (rhythm-aware) models for multi-label alarm classification."""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared CNN feature extractor
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int, pool_freq: bool = True,
                pool_time: bool = False) -> nn.Sequential:
    """Conv + BN + ReLU, optionally followed by pooling.

    *pool_freq* halves the frequency dimension.  *pool_time* halves time.
    By default we only pool frequency to preserve temporal resolution for the GRU.
    """
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool_freq or pool_time:
        k = (2 if pool_freq else 1, 2 if pool_time else 1)
        layers.append(nn.MaxPool2d(k))
    return nn.Sequential(*layers)


class CNNBackbone(nn.Module):
    """CNN that reduces the frequency dimension to 1 while keeping the time axis.

    Input:  (B, 1, N_MELS, T)
    Output: (B, 128, 1, T)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = _conv_block(1, 32, pool_freq=True)
        self.conv2 = _conv_block(32, 64, pool_freq=True)
        self.conv3 = _conv_block(64, 128, pool_freq=True)
        self.conv4 = _conv_block(128, 128, pool_freq=False)

        # Collapse remaining freq bins → 1
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.freq_pool(x)          # (B, 128, 1, T)
        return x


# ---------------------------------------------------------------------------
# CNN-only classifier (original baseline)
# ---------------------------------------------------------------------------

class AlarmClassifier(nn.Module):
    """CNN multi-label classifier (baseline, no temporal modeling).

    Input:  (B, 1, N_MELS, T)
    Output: (B, NUM_CLASSES) raw logits
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = CNNBackbone()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.global_pool(x)        # (B, 128, 1, 1)
        return self.classifier(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs >= threshold).int()


# ---------------------------------------------------------------------------
# CNN + GRU classifier (rhythm-aware)
# ---------------------------------------------------------------------------

class AlarmCRNN(nn.Module):
    """CNN feature extractor + bidirectional GRU for temporal rhythm modeling.

    The CNN collapses the frequency axis; the GRU then learns on-off patterns
    (e.g. ACC's three short beeps vs Caution2's long-on-long-off).

    Input:  (B, 1, N_MELS, T)
    Output: (B, NUM_CLASSES) raw logits
    """

    def __init__(self, num_classes: int, gru_hidden: int = 128, gru_layers: int = 2):
        super().__init__()
        self.backbone = CNNBackbone()   # (B, 128, 1, T)

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if gru_layers > 1 else 0.0,
        )

        gru_out = gru_hidden * 2  # bidirectional

        self.classifier = nn.Sequential(
            nn.Linear(gru_out, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN → (B, 128, 1, T)
        feats = self.backbone(x)

        # Reshape for GRU: (B, 128, 1, T) → (B, T, 128)
        feats = feats.squeeze(2).transpose(1, 2)  # (B, T, 128)

        # GRU → (B, T, gru_hidden*2)
        gru_out, _ = self.gru(feats)

        # Temporal max-pooling: take strongest activation across time
        # This lets the GRU fire on the rhythm pattern regardless of where
        # in the window it occurs.
        pooled, _ = gru_out.max(dim=1)  # (B, gru_hidden*2)

        return self.classifier(pooled)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs >= threshold).int()
