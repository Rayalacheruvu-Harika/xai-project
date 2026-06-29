from pathlib import Path
import random

import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms import Resize
from PIL import Image

from crop_utils import compute_mask_centroid, crop_with_padding


class Stage2Dataset(Dataset):
    def __init__(
        self,
        file_list,
        data_dir,
        mask_dir,
        crop_size=128,
        jitter_px=12
    ):
        self.files = file_list
        self.data_dir = Path(data_dir)
        self.mask_dir = Path(mask_dir)

        self.crop_size = crop_size
        self.jitter_px = jitter_px

        self.resize = Resize((256, 256), antialias=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        # =====================================================
        # Load signal
        # =====================================================
        signal = np.load(self.data_dir / filename).astype(np.float32)

        # Per-channel normalization
        signal_mean = signal.mean(axis=(1, 2), keepdims=True)
        signal_std = signal.std(axis=(1, 2), keepdims=True)
        signal = (signal - signal_mean) / (signal_std + 1e-8)

        # =====================================================
        # Load GT mask
        # =====================================================
        base_name = (
            filename
            .replace("numpy-signal-Vx-", "")
            .replace(".npy", "")
        )

        mask_path = self.mask_dir / f"{base_name}_mask.png"

        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype(np.float32)

        # =====================================================
        # Convert to tensors for resize
        # =====================================================
        signal_t = torch.tensor(signal)
        mask_t = torch.tensor(mask).unsqueeze(0)

        # Resize to 256x256 exactly like Stage 1
        signal_t = self.resize(signal_t)
        mask_t = self.resize(mask_t)

        mask_t = (mask_t > 0.5).float()

        signal = signal_t.numpy()               # [C, 256, 256]
        mask = mask_t.squeeze(0).numpy()        # [256, 256]

        # =====================================================
        # GT centroid + jitter
        # =====================================================
        cx, cy = compute_mask_centroid(mask)

        if self.jitter_px > 0:
            cx += random.randint(-self.jitter_px, self.jitter_px)
            cy += random.randint(-self.jitter_px, self.jitter_px)

        # clamp center into valid image range
        cx = max(0, min(cx, 255))
        cy = max(0, min(cy, 255))

        # =====================================================
        # Crop signal + mask
        # =====================================================
        signal_crop, _ = crop_with_padding(signal, cx, cy, self.crop_size)
        mask_crop, _ = crop_with_padding(mask, cx, cy, self.crop_size)

        signal_crop = torch.tensor(signal_crop).float()
        mask_crop = torch.tensor(mask_crop).unsqueeze(0).float()

        return signal_crop, mask_crop