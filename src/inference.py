from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import DamageDataset
from model import CustomUNet


# =========================================================
# Config
# =========================================================

DEVICE = "cpu"

DATA_DIR = "exp_alu_steel_1_ellip_dam/train_data"
MASK_DIR = "exp_alu_steel_1_ellip_dam/train_masks"

CHECKPOINT_PATH = "checkpoints/best_model.pth"

PRED_DIR = Path("predictions")
PRED_DIR.mkdir(exist_ok=True)


# =========================================================
# Load test file list
# =========================================================

with open("splits/test.txt") as f:
    test_files = [line.strip() for line in f]


# =========================================================
# Dataset
# =========================================================

dataset = DamageDataset(
    test_files,
    DATA_DIR,
    MASK_DIR
)


# =========================================================
# Model
# =========================================================

model = CustomUNet(in_channels=83, out_channels=1).to(DEVICE)

model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=DEVICE)
)

model.eval()


# =========================================================
# Inference
# Save probability maps, not thresholded masks
# =========================================================

with torch.no_grad():
    for idx, filename in enumerate(test_files):
        signal, _ = dataset[idx]

        signal = signal.unsqueeze(0).to(DEVICE)   # [1, C, H, W]

        logits = model(signal)
        probs = torch.sigmoid(logits)

        probs = probs.squeeze().cpu().numpy()     # [H, W]
        probs = np.clip(probs, 0.0, 1.0)

        # Save as grayscale PNG in [0,255]
        prob_uint8 = (probs * 255).astype(np.uint8)

        out_name = f"{Path(filename).stem}_pred.png"
        Image.fromarray(prob_uint8).save(PRED_DIR / out_name)

print("Inference complete")