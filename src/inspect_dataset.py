from pathlib import Path
import numpy as np
import cv2

DATASET_ROOT = "exp_alu_steel_1_ellip_dam"

data_dir = Path(DATASET_ROOT) / "train_data"
mask_dir = Path(DATASET_ROOT) / "train_masks"

data_files = sorted(data_dir.glob("*.npy"))

print(f"Total samples: {len(data_files)}")

# Check first sample
signal = np.load(data_files[0])

print("\nFirst signal:")
print("Shape:", signal.shape)
print("Dtype:", signal.dtype)
print("Min:", signal.min())
print("Max:", signal.max())

# Check corresponding mask
base_name = data_files[0].stem.replace("numpy-signal-Vx-", "")
mask_file = mask_dir / f"{base_name}_mask.png"

mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

print("\nMask:")
print("Shape:", mask.shape)
print("Unique values:", np.unique(mask))