from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize

from model import Stage2UNet
from crop_utils import (
    compute_pred_centroid_from_prob,
    crop_with_padding,
    paste_crop_back
)


# =========================================================
# Config
# =========================================================

DEVICE = "cpu"

DATA_DIR = Path("exp_alu_steel_1_ellip_dam/train_data")
STAGE1_PRED_DIR = Path("predictions")
STAGE2_PRED_DIR = Path("predictions2")
STAGE2_PRED_DIR.mkdir(exist_ok=True)

CROP_SIZE = 128

resize = Resize((256, 256), antialias=True)


# =========================================================
# Test files
# =========================================================

with open("splits/test.txt") as f:
    test_files = [line.strip() for line in f]


# =========================================================
# Load Stage2 model
# =========================================================

model = Stage2UNet(in_channels=83, out_channels=1).to(DEVICE)
model.load_state_dict(
    torch.load("checkpoints_stage2/best_stage2_model.pth", map_location=DEVICE)
)
model.eval()


# =========================================================
# Inference
# =========================================================

with torch.no_grad():
    for filename in test_files:
        # -------------------------------------------------
        # Load original signal
        # -------------------------------------------------
        signal = np.load(DATA_DIR / filename).astype(np.float32)

        # per-channel normalization
        signal_mean = signal.mean(axis=(1, 2), keepdims=True)
        signal_std = signal.std(axis=(1, 2), keepdims=True)
        signal = (signal - signal_mean) / (signal_std + 1e-8)

        signal_t = torch.tensor(signal)
        signal_t = resize(signal_t)
        signal = signal_t.numpy()   # [C, 256, 256]

        # -------------------------------------------------
        # Load Stage1 prediction
        # -------------------------------------------------
        stage1_pred_path = STAGE1_PRED_DIR / f"{Path(filename).stem}_pred.png"

        pred_img = Image.open(stage1_pred_path).convert("L")
        pred = np.array(pred_img).astype(np.float32) / 255.0  # [256, 256]

        # -------------------------------------------------
        # Get coarse centroid from Stage1 prediction
        # -------------------------------------------------
        cx, cy = compute_pred_centroid_from_prob(pred, threshold=0.5)

        # -------------------------------------------------
        # Crop signal around Stage1 center
        # -------------------------------------------------
        signal_crop, meta = crop_with_padding(signal, cx, cy, CROP_SIZE)

        signal_crop_t = torch.tensor(signal_crop).unsqueeze(0).float().to(DEVICE)

        # -------------------------------------------------
        # Stage2 prediction on crop
        # -------------------------------------------------
        logits = model(signal_crop_t)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()   # [128, 128]

        # -------------------------------------------------
        # Paste back to full 256x256
        # -------------------------------------------------
        full_pred = paste_crop_back(prob, meta)

        # save probability map as 8-bit png
        full_pred_uint8 = (full_pred * 255).astype(np.uint8)

        out_path = STAGE2_PRED_DIR / f"{Path(filename).stem}_pred.png"
        Image.fromarray(full_pred_uint8).save(out_path)

print("Stage2 inference complete")