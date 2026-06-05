# src/visualize_predictions.py

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

pred_dir = Path("predictions")
gt_dir = Path("generated_test_masks")

for pred_file in pred_dir.glob("*_pred.png"):

    gt_name = (
        pred_file.name
        .replace("numpy-signal-Vx-", "")
        .replace("_pred.png", "_mask.png")
    )

    gt_file = gt_dir / gt_name

    pred = np.array(
        Image.open(pred_file).convert("L")
    )

    gt = np.array(
        Image.open(gt_file).convert("L")
        .resize((256,256))
    )

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(gt)
    plt.title("Ground Truth")

    plt.subplot(1,2,2)
    plt.imshow(pred)
    plt.title("Prediction")

    plt.show()