
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

SHAP_DIR = Path("shap_results")
PRED_DIR = Path("predictions")
GT_DIR = Path("generated_test_masks")
OUT_DIR = Path("shap_figures")
OUT_DIR.mkdir(exist_ok=True)

files = sorted(SHAP_DIR.glob("*_shap.npy"))
if not files:
    raise FileNotFoundError("No SHAP files found.")

channel_scores = []
alignment_scores = []
peak_distances = []

# ==========================================================
# Centroid Function
# ==========================================================

def centroid(mask):

    y, x = np.where(mask)

    if len(x) == 0:
        return np.array([0.0, 0.0])

    return np.array([
        x.mean(),
        y.mean()
    ])

for f in files:

    shap_values = np.load(f)
    shap_values = shap_values[0]
    shap_values = shap_values[..., 0]

    spatial = np.abs(shap_values).mean(axis=0)

    sample_name = f.stem.replace("_shap", "")

    pred_file = PRED_DIR / f"{sample_name}_pred.png"

    gt_name = (
        f"{sample_name}_mask.png"
        .replace("numpy-signal-Vx-", "")
    )

    gt_file = GT_DIR / gt_name

    if pred_file.exists():
        pred = np.array(Image.open(pred_file).convert("L"))
    else:
        pred = np.zeros((256, 256), dtype=np.uint8)

    if gt_file.exists():
        gt = np.array(
            Image.open(gt_file)
            .convert("L")
            .resize((256, 256))
        )
    else:
        gt = np.zeros((256, 256), dtype=np.uint8)

    gt_mask = gt > 0

    spatial_norm = spatial / (spatial.max() + 1e-8)

   # ------------------------------------------------------
    # Alignment (unchanged)
    # ------------------------------------------------------

    shap_alignment = (
        spatial_norm[gt_mask].sum()
    ) / (
        spatial_norm.sum() + 1e-8
    )

    alignment_scores.append(shap_alignment)

    # ------------------------------------------------------
    # Peak Distance
    # ------------------------------------------------------

    gt_centroid = centroid(gt_mask)

    peak_y, peak_x = np.unravel_index(
        np.argmax(spatial_norm),
        spatial_norm.shape
    )

    peak = np.array([
        peak_x,
        peak_y
    ])

    peak_distance = np.linalg.norm(
        peak - gt_centroid
    )

    peak_distances.append(peak_distance)

    print("=" * 60)
    print(sample_name)
    print(f"SHAP Alignment    : {shap_alignment:.4f}")
    print(f"SHAP Peak Distance: {peak_distance:.2f} pixels")
    print("=" * 60)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(gt, cmap="gray")
    ax[0].set_title("Ground Truth")
    ax[0].axis("off")

    ax[1].imshow(pred, cmap="gray")
    ax[1].set_title("Prediction")
    ax[1].axis("off")

    im = ax[2].imshow(spatial, cmap="jet")
    ax[2].set_title("SHAP")
    ax[2].axis("off")

    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04, label="|SHAP|")

    plt.suptitle(sample_name)
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"{sample_name}_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    channel_scores.append(
        np.abs(shap_values).mean(axis=(1, 2))
    )

channel_scores = np.array(channel_scores)
mean_channel = channel_scores.mean(axis=0)

top = np.argsort(mean_channel)[::-1][:10]

plt.figure(figsize=(8, 5))
plt.bar(range(len(top)), mean_channel[top])
plt.xticks(range(len(top)), [f"Ch {i}" for i in top], rotation=45)
plt.ylabel("Mean |SHAP|")
plt.xlabel("Channel")
plt.title("Top 10 Most Important Channels")
plt.tight_layout()
plt.savefig(OUT_DIR / "top10_channels.png", dpi=300)
plt.close()

np.savetxt(
    OUT_DIR / "channel_importance.csv",
    np.column_stack((np.arange(len(mean_channel)), mean_channel)),
    delimiter=",",
    header="channel,importance",
    comments=""
)

print("\n==============================")
print("SHAP ALIGNMENT SUMMARY")
print("==============================")

for f, score in zip(files, alignment_scores):
    print(f"{f.stem.replace('_shap','')}: {score:.4f}")

print(f"\nMean Alignment : {np.mean(alignment_scores):.4f}")
print(f"Std Alignment  : {np.std(alignment_scores):.4f}")
print(f"\nMean Peak Distance : {np.mean(peak_distances):.2f} pixels")
print(f"Std Peak Distance  : {np.std(peak_distances):.2f} pixels")

print("\nFinished.")
print("Results saved to:", OUT_DIR)
