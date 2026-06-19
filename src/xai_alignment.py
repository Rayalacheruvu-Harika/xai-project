import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image


GT_DIR = Path("generated_test_masks")

GRADCAM_DIR = Path("gradcam_maps")
IG_DIR = Path("ig_maps")

results = []


def centroid(mask):

    y, x = np.where(mask > 0)

    if len(x) == 0:
        return np.array([0, 0])

    return np.array([
        x.mean(),
        y.mean()
    ])


for gt_file in GT_DIR.glob("*_mask.png"):

    name = gt_file.stem.replace(
    "_mask",
    ""
)

    gradcam_file = (
        GRADCAM_DIR /
        f"numpy-signal-Vx-{name}_gradcam.png"
    )

    ig_file = (
        IG_DIR /
        f"numpy-signal-Vx-{name}_ig.png"
    )

    if (
        not gradcam_file.exists()
        or
        not ig_file.exists()
    ):
        continue

    gt = np.array(
        Image.open(gt_file)
        .convert("L")
        .resize((256, 256))
    ) > 0

    gt_centroid = centroid(gt)

    gradcam = np.array(
        Image.open(gradcam_file)
        .convert("L")
    ).astype(np.float32)

    ig = np.array(
        Image.open(ig_file)
        .convert("L")
    ).astype(np.float32)

    gradcam /= (
        gradcam.max() + 1e-8
    )

    ig /= (
        ig.max() + 1e-8
    )

    gradcam_alignment = (
        gradcam[gt].sum()
    ) / (
        gradcam.sum() + 1e-8
    )

    ig_alignment = (
        ig[gt].sum()
    ) / (
        ig.sum() + 1e-8
    )

    gy, gx = np.unravel_index(
        gradcam.argmax(),
        gradcam.shape
    )

    iy, ix = np.unravel_index(
        ig.argmax(),
        ig.shape
    )

    gradcam_peak = np.array([
        gx,
        gy
    ])

    ig_peak = np.array([
        ix,
        iy
    ])

    gradcam_distance = np.linalg.norm(
        gradcam_peak - gt_centroid
    )

    ig_distance = np.linalg.norm(
        ig_peak - gt_centroid
    )

    results.append([
        name,
        gradcam_alignment,
        gradcam_distance,
        ig_alignment,
        ig_distance
    ])


df = pd.DataFrame(
    results,
    columns=[
        "file",
        "gradcam_alignment",
        "gradcam_peak_distance",
        "ig_alignment",
        "ig_peak_distance"
    ]
)

print(df)

print("\n===== SUMMARY =====")

print(
    "GradCAM Alignment:",
    df["gradcam_alignment"].mean()
)

print(
    "GradCAM Distance:",
    df["gradcam_peak_distance"].mean()
)

print(
    "IG Alignment:",
    df["ig_alignment"].mean()
)

print(
    "IG Distance:",
    df["ig_peak_distance"].mean()
)

Path("results").mkdir(
    exist_ok=True
)

df.to_csv(
    "results/xai_metrics.csv",
    index=False
)

print(
    "\nSaved results/xai_metrics.csv"
)