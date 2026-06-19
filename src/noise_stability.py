import torch
import numpy as np
import pandas as pd

from pathlib import Path

from dataset import DamageDataset
from model import CustomUNet


DEVICE = "cpu"

NOISE_LEVELS = [
    0.00,
    0.01,
    0.05
]


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def dice_score(pred, gt):

    intersection = (
        pred * gt
    ).sum()

    return (
        2 * intersection
    ) / (
        pred.sum() +
        gt.sum() +
        1e-8
    )


def iou_score(pred, gt):

    intersection = (
        pred * gt
    ).sum()

    union = (
        pred + gt
    ) > 0

    return (
        intersection
    ) / (
        union.sum() +
        1e-8
    )


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = CustomUNet(
    in_channels=83,
    out_channels=1
)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=DEVICE
    )
)

model.eval()

# --------------------------------------------------
# Dataset
# --------------------------------------------------

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

results = []

# --------------------------------------------------
# Loop
# --------------------------------------------------

for noise_std in NOISE_LEVELS:

    all_dice = []
    all_iou = []

    for idx in range(len(dataset)):

        signal, gt_mask = dataset[idx]

        signal_np = signal.numpy()

        noise = np.random.normal(
            0,
            noise_std,
            signal_np.shape
        )

        noisy_signal = (
            signal_np + noise
        )

        input_tensor = torch.tensor(
            noisy_signal,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            pred = torch.sigmoid(
                model(input_tensor)
            )

        pred = (
            pred.squeeze()
            .numpy()
        )

        pred = (
            pred > 0.5
        ).astype(np.float32)

        gt = (
            gt_mask.squeeze()
            .numpy()
        )

        dice = dice_score(
            pred,
            gt
        )

        iou = iou_score(
            pred,
            gt
        )

        all_dice.append(
            dice
        )

        all_iou.append(
            iou
        )

    results.append([
        noise_std,
        np.mean(all_dice),
        np.std(all_dice),
        np.mean(all_iou),
        np.std(all_iou)
    ])

# --------------------------------------------------
# Save
# --------------------------------------------------

df = pd.DataFrame(
    results,
    columns=[
        "noise_std",
        "dice_mean",
        "dice_std",
        "iou_mean",
        "iou_std"
    ]
)

print(df)

Path("results").mkdir(
    exist_ok=True
)

df.to_csv(
    "results/noise_stability.csv",
    index=False
)

print(
    "\nSaved: results/noise_stability.csv"
)