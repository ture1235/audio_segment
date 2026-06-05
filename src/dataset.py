"""PyTorch Dataset for on-the-fly synthetic sample generation."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import generate_sample
from .features import audio_to_melspec


class AlarmDataset(Dataset):
    """Dataset that dynamically generates synthetic alarm+noise samples.

    Each __getitem__ call mixes alarms and noise on-the-fly, so the
    training data is effectively infinite and never repeats identically.

    Args:
        alarms: dict[int, np.ndarray]  alarm_index -> audio array
        noise_list: list[np.ndarray]   noise audio arrays
        num_classes: int               number of alarm types
        alarm_indices: list[int]|None  restrict to these alarm indices (for val split)
        length: int                    virtual dataset length (one epoch)
    """

    def __init__(
        self,
        alarms: dict,
        noise_list: list,
        num_classes: int,
        alarm_indices: list[int] | None = None,
        length: int = 8000,
    ):
        self.alarms = alarms
        self.noise_list = noise_list
        self.num_classes = num_classes
        self.alarm_indices = alarm_indices
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio, label = generate_sample(
            self.alarms,
            self.noise_list,
            self.alarm_indices,
        )
        melspec = audio_to_melspec(audio)
        return torch.from_numpy(melspec), torch.from_numpy(label)
