import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image

pred_dir = Path("predictions")
gt_dir = Path("generated_test_masks")

results = []

THRESHOLD = 0.5

for pred_file in pred_dir.glob("*_pred.png"):

    base_name = (
        pred_file.name
        .replace("numpy-signal-Vx-", "")
        .replace("_pred.png", "_mask.png")
    )

    gt_file = gt_dir / base_name

    # --------------------------
    # Load prediction
    # --------------------------

    pred = Image.open(pred_file).convert("L")

    pred = np.array(pred).astype(np.float32)

    pred = pred / 255.0

    pred = pred > THRESHOLD

    # --------------------------
    # Load ground truth
    # --------------------------

    gt = Image.open(gt_file).convert("L")

    gt = gt.resize(
        (256, 256),
        Image.NEAREST
    )

    gt = np.array(gt) > 0

    # --------------------------
    # Metrics
    # --------------------------

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    iou = (
        intersection /
        (union + 1e-8)
    )

    dice = (
        2 * intersection
    ) / (
        pred.sum() +
        gt.sum() +
        1e-8
    )

    results.append([
        pred_file.name,
        dice,
        iou
    ])

# --------------------------
# Results table
# --------------------------

df = pd.DataFrame(
    results,
    columns=[
        "file",
        "dice",
        "iou"
    ]
)

print(df)

print(
    "\nMean Dice:",
    df["dice"].mean()
)

print(
    "Mean IoU:",
    df["iou"].mean()
)

# --------------------------
# Save results
# --------------------------

Path("results").mkdir(
    exist_ok=True
)

df.to_csv(
    "results/metrics.csv",
    index=False
)