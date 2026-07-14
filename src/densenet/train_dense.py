from pathlib import Path
import random
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from dataset import DamageDataset
from model_dense import DenseUNet


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

EPOCHS = 30

LR = 1e-4

BATCH_SIZE = 1

WEIGHT_DECAY = 1e-4

POS_WEIGHT = 10.0

VAL_THRESHOLD = 0.5

EARLY_STOPPING_PATIENCE = 10


Path("checkpoints_dense").mkdir(exist_ok=True)


# =========================================================
# Train / Validation split
# =========================================================

with open("splits/train.txt") as f:
    train_files = [
        line.strip()
        for line in f
    ]


with open("splits/val.txt") as f:
    val_files = [
        line.strip()
        for line in f
    ]


print(f"Train images : {len(train_files)}")
print(f"Validation   : {len(val_files)}")


# =========================================================
# Dataset
# =========================================================

train_dataset = DamageDataset(
    train_files,
    DATA_DIR,
    MASK_DIR
)

val_dataset = DamageDataset(
    val_files,
    DATA_DIR,
    MASK_DIR
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

model = DenseUNet(
    in_channels=83,
    out_channels=1
).to(DEVICE)


# =========================================================
# Optimizer
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


# =========================================================
# Cyclic Learning Rate
# =========================================================

scheduler = torch.optim.lr_scheduler.CyclicLR(

    optimizer,

    base_lr=1e-5,

    max_lr=5e-4,

    step_size_up=len(train_loader) * 4,

    mode="triangular2",

    cycle_momentum=False
)


# =========================================================
# Loss
# =========================================================

pos_weight = torch.tensor(
    [POS_WEIGHT],
    device=DEVICE
)

bce_loss = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


def dice_loss(
    logits,
    targets
):

    probs = torch.sigmoid(logits)

    smooth = 1e-6

    probs = probs.view(
        probs.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
    )

    intersection = (
        probs * targets
    ).sum(dim=1)

    union = (
        probs.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        union + smooth
    )

    return (
        1.0 - dice
    ).mean()


def compute_loss(
    logits,
    targets
):

    return (
        bce_loss(
            logits,
            targets
        )
        +
        dice_loss(
            logits,
            targets
        )
    )


# =========================================================
# Validation Dice
# =========================================================

def dice_score(
    logits,
    targets,
    threshold=0.5
):

    probs = torch.sigmoid(logits)

    preds = (
        probs > threshold
    ).float()

    preds = preds.view(
        preds.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
    )

    intersection = (
        preds * targets
    ).sum(dim=1)

    union = (
        preds.sum(dim=1)
        +
        targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        union + 1e-6
    )

    return dice.mean().item()


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

            loss = compute_loss(
                logits,
                mask
            )

            dice = dice_score(
                logits,
                mask,
                VAL_THRESHOLD
            )

            total_loss += loss.item()

            total_dice += dice

    mean_loss = (
        total_loss
        / len(val_loader)
    )

    mean_dice = (
        total_dice
        / len(val_loader)
    )

    return mean_loss, mean_dice

# =========================================================
# Training Loop
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

        loss = compute_loss(
            logits,
            mask
        )

        loss.backward()

        optimizer.step()

        # -----------------------------------------
        # Cyclic Learning Rate updates EVERY BATCH
        # -----------------------------------------
        scheduler.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    val_loss, val_dice = validate()

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1:03d} | "
        f"Train Loss {train_loss:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Dice {val_dice:.4f} | "
        f"LR {current_lr:.2e}"
    )

    # -----------------------------------------
    # Save Best Model
    # -----------------------------------------

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        epochs_without_improvement = 0

        torch.save(

            model.state_dict(),

            "checkpoints_dense/best_dense_model.pth"

        )

        print(
            ">>> Best DenseUNet model saved."
        )

    else:

        epochs_without_improvement += 1

    # -----------------------------------------
    # Early Stopping
    # -----------------------------------------

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# =========================================================
# Training Finished
# =========================================================

print("\n===================================")
print("Training Complete")
print("===================================")

print(
    f"Best Validation Dice : "
    f"{best_val_dice:.4f}"
)

print(
    "Model saved at:\n"
    "checkpoints_dense/best_dense_model.pth"
)
