import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image


# ==================================================
# Pixel to mm conversion
# ==================================================
# Example:
# if plate width = 500 mm and image width = 256 px
# PIXEL_SIZE_MM = 500 / 256 = 1.953125

PIXEL_SIZE_MM = 1.953125
THRESHOLD = 0.5


# ==================================================
# Directories
# ==================================================

PRED_DIR = Path("predictions2")
GT_DIR = Path("generated_test_masks")

results = []


# ==================================================
# Evaluate all test predictions
# ==================================================

for pred_file in PRED_DIR.glob("*_pred.png"):

    base_name = (
        pred_file.name
        .replace("numpy-signal-Vx-", "")
        .replace("_pred.png", "_mask.png")
    )

    gt_file = GT_DIR / base_name

    if not gt_file.exists():
        print(f"Missing GT: {gt_file}")
        continue

    # ----------------------------------------------
    # Load prediction
    # Same thresholding as evaluate.py
    # ----------------------------------------------

    pred = Image.open(pred_file).convert("L")
    pred = np.array(pred).astype(np.float32)
    pred = pred / 255.0
    pred = pred > THRESHOLD

    # ----------------------------------------------
    # Load ground truth
    # ----------------------------------------------

    gt = Image.open(gt_file).convert("L")
    gt = gt.resize((256, 256), Image.NEAREST)
    gt = np.array(gt) > 0

    # ----------------------------------------------
    # Dice and IoU
    # ----------------------------------------------

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    dice = (
        2 * intersection
    ) / (
        pred.sum() +
        gt.sum() +
        1e-8
    )

    iou = (
        intersection
    ) / (
        union +
        1e-8
    )

    # ----------------------------------------------
    # F1 Score
    # ----------------------------------------------

    tp = intersection

    fp = np.logical_and(
        pred == 1,
        gt == 0
    ).sum()

    fn = np.logical_and(
        pred == 0,
        gt == 1
    ).sum()

    f1 = (
        2 * tp
    ) / (
        2 * tp + fp + fn + 1e-8
    )

    # ----------------------------------------------
    # Centroids
    # ----------------------------------------------

    pred_y, pred_x = np.where(pred)
    gt_y, gt_x = np.where(gt)

    if len(pred_x) > 0:
        pred_centroid = np.array([
            pred_x.mean(),
            pred_y.mean()
        ])
    else:
        pred_centroid = np.array([
            0,
            0
        ])

    gt_centroid = np.array([
        gt_x.mean(),
        gt_y.mean()
    ])

    # ----------------------------------------------
    # Localization error
    # ----------------------------------------------

    centroid_error_px = np.linalg.norm(
        pred_centroid - gt_centroid
    )

    centroid_error_mm = (
        centroid_error_px *
        PIXEL_SIZE_MM
    )

    # ----------------------------------------------
    # Store results
    # ----------------------------------------------

    results.append([
        pred_file.name,
        dice,
        f1,
        iou,
        centroid_error_px,
        centroid_error_mm
    ])


# ==================================================
# DataFrame
# ==================================================

df = pd.DataFrame(
    results,
    columns=[
        "file",
        "dice",
        "f1",
        "iou",
        "centroid_error_px",
        "centroid_error_mm"
    ]
)

print(df)

# ==================================================
# Summary
# ==================================================

print("\n========== SUMMARY ==========")

print(
    f"Dice: "
    f"{df['dice'].mean():.4f} ± "
    f"{df['dice'].std():.4f}"
)

print(
    f"F1:   "
    f"{df['f1'].mean():.4f} ± "
    f"{df['f1'].std():.4f}"
)

print(
    f"IoU: "
    f"{df['iou'].mean():.4f} ± "
    f"{df['iou'].std():.4f}"
)

print(
    f"Centroid Error: "
    f"{df['centroid_error_px'].mean():.2f} ± "
    f"{df['centroid_error_px'].std():.2f} pixels"
)

print(
    f"Centroid Error: "
    f"{df['centroid_error_mm'].mean():.2f} ± "
    f"{df['centroid_error_mm'].std():.2f} mm"
)

# ==================================================
# Save CSV
# ==================================================

Path("results").mkdir(exist_ok=True)

df.to_csv(
    "results/test_metrics.csv",
    index=False
)

print("\nSaved: results/test_metrics.csv")