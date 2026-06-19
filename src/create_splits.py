from pathlib import Path
import random
import json

random.seed(42)

# ==================================================
# Load files
# ==================================================

DATA_DIR = Path(
    "exp_alu_steel_1_ellip_dam/train_data"
)

files = sorted(DATA_DIR.glob("*.npy"))
files = [f.name for f in files]

random.shuffle(files)

# ==================================================
# Create splits
# ==================================================

train = files[:28]
val = files[28:34]
test = files[34:]

# ==================================================
# Verify overlap
# ==================================================

train_set = set(train)
val_set = set(val)
test_set = set(test)

train_val_overlap = train_set.intersection(val_set)
train_test_overlap = train_set.intersection(test_set)
val_test_overlap = val_set.intersection(test_set)

print("\n========== OVERLAP CHECK ==========")

print(
    f"Train-Val Overlap: "
    f"{len(train_val_overlap)}"
)

print(
    f"Train-Test Overlap: "
    f"{len(train_test_overlap)}"
)

print(
    f"Val-Test Overlap: "
    f"{len(val_test_overlap)}"
)

# ==================================================
# Save TXT splits
# ==================================================

Path("splits").mkdir(
    exist_ok=True
)

with open(
    "splits/train.txt",
    "w"
) as f:
    f.write("\n".join(train))

with open(
    "splits/val.txt",
    "w"
) as f:
    f.write("\n".join(val))

with open(
    "splits/test.txt",
    "w"
) as f:
    f.write("\n".join(test))

# ==================================================
# Save JSON split file
# ==================================================

split_info = {
    "random_seed": 42,
    "train": train,
    "validation": val,
    "test": test,
    "overlap_check": {
        "train_validation_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "validation_test_overlap": len(val_test_overlap)
    }
}

with open(
    "splits/splits.json",
    "w"
) as f:
    json.dump(
        split_info,
        f,
        indent=4
    )

# ==================================================
# Summary
# ==================================================

print("\n========== SPLIT SUMMARY ==========")

print(f"Train: {len(train)}")
print(f"Val:   {len(val)}")
print(f"Test:  {len(test)}")

print(
    "\nSaved:"
    "\n - splits/train.txt"
    "\n - splits/val.txt"
    "\n - splits/test.txt"
    "\n - splits/splits.json"
)