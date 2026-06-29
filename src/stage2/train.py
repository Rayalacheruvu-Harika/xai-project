from pathlib import Path
import random
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from stage2_dataset import Stage2Dataset
from model import Stage2UNet


# =========================================================
# Reproducibility
# =========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# Configuration
# =========================================================

DATA_DIR = "exp_alu_steel_1_ellip_dam/train_data"
MASK_DIR = "exp_alu_steel_1_ellip_dam/train_masks"

DEVICE = "cpu"

EPOCHS = 50
LR = 1e-4
BATCH_SIZE = 1
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 8.0

CROP_SIZE = 128
JITTER_PX = 12

VAL_THRESHOLD = 0.5
EARLY_STOPPING_PATIENCE = 12

Path("checkpoints_stage2").mkdir(exist_ok=True)


# =========================================================
# Read splits
# =========================================================

with open("splits/train.txt") as f:
    train_files = [line.strip() for line in f]

with open("splits/val.txt") as f:
    val_files = [line.strip() for line in f]

print(f"Stage2 Train samples: {len(train_files)}")
print(f"Stage2 Val samples:   {len(val_files)}")


# =========================================================
# Dataset / loaders
# =========================================================

train_dataset = Stage2Dataset(
    train_files,
    DATA_DIR,
    MASK_DIR,
    crop_size=CROP_SIZE,
    jitter_px=JITTER_PX
)

val_dataset = Stage2Dataset(
    val_files,
    DATA_DIR,
    MASK_DIR,
    crop_size=CROP_SIZE,
    jitter_px=0
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================================================
# Model
# =========================================================

model = Stage2UNet(in_channels=83, out_channels=1).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)
bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# =========================================================
# Loss / metrics
# =========================================================

def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    smooth = 1e-6

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def dice_score_from_logits(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item()


def compute_loss(logits, targets):
    return bce_loss(logits, targets) + dice_loss(logits, targets)


# =========================================================
# Validation
# =========================================================

def validate():
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for signal, mask in val_loader:
            signal = signal.to(DEVICE)
            mask = mask.to(DEVICE)

            logits = model(signal)

            loss = compute_loss(logits, mask)
            dice = dice_score_from_logits(logits, mask, threshold=VAL_THRESHOLD)

            total_loss += loss.item()
            total_dice += dice

    mean_loss = total_loss / len(val_loader)
    mean_dice = total_dice / len(val_loader)

    return mean_loss, mean_dice


# =========================================================
# Training loop
# =========================================================

best_val_dice = -1.0
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for signal, mask in train_loader:
        signal = signal.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()

        logits = model(signal)
        loss = compute_loss(logits, mask)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss, val_dice = validate()

    print(
        f"Stage2 Epoch {epoch + 1:02d} | "
        f"TrainLoss {train_loss:.4f} | "
        f"ValLoss {val_loss:.4f} | "
        f"ValDice {val_dice:.4f}"
    )

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        epochs_without_improvement = 0

        torch.save(model.state_dict(), "checkpoints_stage2/best_stage2_model.pth")
        print("  -> Best Stage2 model saved")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(
            f"Early stopping Stage2 after {epoch + 1} epochs. "
            f"Best Val Dice: {best_val_dice:.4f}"
        )
        break

print(f"\nStage2 training complete. Best validation Dice: {best_val_dice:.4f}")