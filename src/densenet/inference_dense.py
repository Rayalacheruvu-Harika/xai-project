from pathlib import Path

import torch
import numpy as np

from PIL import Image

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from dataset import DamageDataset
from model_dense import DenseUNet


# =========================================================
# Configuration
# =========================================================

DEVICE = "cpu"

DATA_DIR = "exp_alu_steel_1_ellip_dam/train_data"
MASK_DIR = "exp_alu_steel_1_ellip_dam/train_masks"

MODEL_PATH = "checkpoints_dense/best_dense_model.pth"

OUTPUT_DIR = Path("predictions_dense")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# Test files
# =========================================================

with open("splits/test.txt") as f:
    test_files = [
        line.strip()
        for line in f
    ]


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

model = DenseUNet(
    in_channels=83,
    out_channels=1
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()


# =========================================================
# Inference
# =========================================================

with torch.no_grad():

    for idx, filename in enumerate(test_files):

        signal, _ = dataset[idx]

        signal = signal.unsqueeze(0).to(DEVICE)

        logits = model(signal)

        probs = torch.sigmoid(logits)

        pred = probs.squeeze().cpu().numpy()

        pred = (pred * 255).astype(np.uint8)

        Image.fromarray(pred).save(

            OUTPUT_DIR /

            f"{Path(filename).stem}_pred.png"

        )

print("\n===================================")
print("DenseUNet Inference Complete")
print("===================================")
print(f"Predictions saved to: {OUTPUT_DIR}")