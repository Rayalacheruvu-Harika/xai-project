from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms import Resize

from PIL import Image


class DamageDataset(Dataset):

    def __init__(self, file_list, data_dir, mask_dir):

        self.files = file_list

        self.data_dir = Path(data_dir)
        self.mask_dir = Path(mask_dir)

        self.resize = Resize(
            (256, 256),
            antialias=True
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        # ----------------------------
        # Load signal
        # ----------------------------
        signal = np.load(
            self.data_dir / filename
        ).astype(np.float32)

        # Normalize
        signal_mean = signal.mean(axis=(1, 2), keepdims=True)
        signal_std  = signal.std(axis=(1, 2), keepdims=True)
        signal = (signal - signal_mean) / (signal_std + 1e-8)

        # ----------------------------
        # Load mask
        # ----------------------------
        base_name = (
            filename
            .replace("numpy-signal-Vx-", "")
            .replace(".npy", "")
        )

        mask_path = (
            self.mask_dir /
            f"{base_name}_mask.png"
        )

        mask = np.array(
            Image.open(mask_path)
            .convert("L")
        )

        mask = (mask > 0).astype(np.float32)

        # ----------------------------
        # Convert to tensors
        # ----------------------------
        signal = torch.tensor(signal)

        mask = torch.tensor(mask).unsqueeze(0)

        # ----------------------------
        # Resize
        # ----------------------------
        signal = self.resize(signal)

        mask = self.resize(mask)

        # Make mask binary again
        mask = (mask > 0.5).float()

        return signal.float(), mask.float()