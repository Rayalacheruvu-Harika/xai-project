from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import DamageDataset
from model import get_model


# ------------------------
# Configuration
# ------------------------

DATA_DIR = "exp_alu_steel_1_ellip_dam/train_data"
MASK_DIR = "exp_alu_steel_1_ellip_dam/train_masks"

DEVICE = "cpu"

EPOCHS = 50
BATCH_SIZE = 1
LR = 1e-4

Path("checkpoints").mkdir(exist_ok=True)


# ------------------------
# Read splits
# ------------------------

with open("splits/train.txt") as f:
    train_files = [
        line.strip()
        for line in f.readlines()
    ]

with open("splits/val.txt") as f:
    val_files = [
        line.strip()
        for line in f.readlines()
    ]


# ------------------------
# Datasets
# ------------------------

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
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ------------------------
# Model
# ------------------------

model = get_model().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

bce_loss = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([20.0])
)


# ------------------------
# Dice Loss
# ------------------------

def dice_loss(pred, target):

    pred = torch.sigmoid(pred)

    smooth = 1e-6

    intersection = (
        pred * target
    ).sum()

    union = (
        pred + target
    ).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        union + smooth
    )

    return 1 - dice


# ------------------------
# Validation
# ------------------------

def validate():

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for signal, mask in val_loader:

            signal = signal.to(DEVICE)
            mask = mask.to(DEVICE)

            pred = model(signal)

            loss = (
                bce_loss(pred, mask)
                +
                dice_loss(pred, mask)
            )

            total_loss += loss.item()

    return total_loss / len(val_loader)


# ------------------------
# Training Loop
# ------------------------

best_val = float("inf")

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for signal, mask in train_loader:

        signal = signal.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()

        pred = model(signal)

        loss = (
            bce_loss(pred, mask)
            +
            dice_loss(pred, mask)
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    val_loss = validate()

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train {train_loss:.4f} | "
        f"Val {val_loss:.4f}"
    )

    if val_loss < best_val:

        best_val = val_loss

        torch.save(
            model.state_dict(),
            "checkpoints/best_model.pth"
        )

        print("Best model saved")