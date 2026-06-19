from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


SAVE_DIR = Path(
    "results/case_study"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

cases = [

    "train_damage_ellipse_xy_275_181_ab_75_14_th_46",
    "train_damage_ellipse_xy_255_418_ab_50_13_th_178",
    "train_damage_ellipse_xy_110_285_ab_58_12_th_180",
    "train_damage_ellipse_xy_178_404_ab_84_13_th_83",
    "train_damage_ellipse_xy_175_320_ab_60_9_th_136"
]

for case in cases:

    gt = np.array(
        Image.open(
            f"generated_test_masks/{case}_mask.png"
        )
    )

    pred = np.array(
        Image.open(
            f"predictions/numpy-signal-Vx-{case}_pred.png"
        )
    )

    gradcam = np.array(
        Image.open(
            f"gradcam_maps/numpy-signal-Vx-{case}_gradcam.png"
        )
    )

    ig = np.array(
        Image.open(
            f"ig_maps/numpy-signal-Vx-{case}_ig.png"
        )
    )

    plt.figure(
        figsize=(12,3)
    )

    plt.subplot(1,4,1)
    plt.imshow(gt,cmap="gray")
    plt.title("GT")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(pred,cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(gradcam,cmap="jet")
    plt.title("GradCAM")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(ig,cmap="jet")
    plt.title("IG")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        SAVE_DIR /
        f"{case}.png"
    )

    plt.close()

print(
    "Case study figures saved."
)