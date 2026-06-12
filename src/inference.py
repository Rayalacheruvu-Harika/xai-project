import torch
import numpy as np

from pathlib import Path
from PIL import Image

from dataset import DamageDataset
from model import CustomUNet


DEVICE = "cpu"

PRED_DIR = Path("predictions")
PRED_DIR.mkdir(exist_ok=True)

with open("splits/test.txt") as f:
    test_files = [
        line.strip()
        for line in f
    ]

dataset = DamageDataset(
    test_files,
    "exp_alu_steel_1_ellip_dam/train_data",
    "exp_alu_steel_1_ellip_dam/train_masks"
)

model = CustomUNet(
    in_channels=83,
    out_channels=1
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=DEVICE
    )
)

model.eval()

with torch.no_grad():

    for idx, filename in enumerate(test_files):

        signal, _ = dataset[idx]

        signal = signal.unsqueeze(0)

        pred = model(signal)

        pred = torch.sigmoid(pred)

        pred = pred.squeeze().cpu().numpy()

        pred = (pred * 255).astype(np.uint8)

        Image.fromarray(pred).save(
            PRED_DIR /
            f"{Path(filename).stem}_pred.png"
        )

print("Inference complete")