import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image


GT_DIR = Path("generated_test_masks")
PRED_DIR = Path("predictions")

GRADCAM_DIR = Path("gradcam_maps")
IG_DIR = Path("ig_maps")

SAVE_DIR = Path("results/overlays")
SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

for gt_file in GT_DIR.glob("*_mask.png"):

    name = gt_file.stem.replace(
        "_mask",
        ""
    )

    pred_file = (
        PRED_DIR /
        f"numpy-signal-Vx-{name}_pred.png"
    )

    gradcam_file = (
        GRADCAM_DIR /
        f"numpy-signal-Vx-{name}_gradcam.png"
    )

    ig_file = (
        IG_DIR /
        f"numpy-signal-Vx-{name}_ig.png"
    )

    gt = np.array(
        Image.open(gt_file)
        .convert("L")
        .resize((256,256))
    )

    pred = np.array(
        Image.open(pred_file)
        .convert("L")
    )

    gradcam = np.array(
        Image.open(gradcam_file)
        .convert("L")
    )

    ig = np.array(
        Image.open(ig_file)
        .convert("L")
    )

    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    plt.imshow(gt,cmap="gray")
    plt.imshow(
        gradcam,
        cmap="jet",
        alpha=0.5
    )
    plt.title("GradCAM + GT")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.imshow(pred,cmap="gray")
    plt.imshow(
        gradcam,
        cmap="jet",
        alpha=0.5
    )
    plt.title("GradCAM + Prediction")
    plt.axis("off")

    plt.subplot(2,2,3)
    plt.imshow(gt,cmap="gray")
    plt.imshow(
        ig,
        cmap="jet",
        alpha=0.5
    )
    plt.title("IG + GT")
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.imshow(pred,cmap="gray")
    plt.imshow(
        ig,
        cmap="jet",
        alpha=0.5
    )
    plt.title("IG + Prediction")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        SAVE_DIR /
        f"{name}_overlay.png"
    )

    plt.close()

print("Overlay figures saved")