import torch
import numpy as np
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients

from dataset import DamageDataset
from model import CustomUNet


# =====================================================
# Segmentation Wrapper
# =====================================================

class SegmentationWrapper(torch.nn.Module):

    def __init__(self, model, mask):
        super().__init__()

        self.model = model
        self.mask = mask

    def forward(self, x):

        output = self.model(x)

        score = (
            output.squeeze() *
            self.mask
        ).sum()

        return score.view(1)


# =====================================================
# Device
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# Load model
# =====================================================

model = CustomUNet(
    in_channels=83,
    out_channels=1
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=DEVICE
    )
)

model.eval()

# =====================================================
# Load test data
# =====================================================

with open("splits/test.txt") as f:
    test_files = [
        line.strip()
        for line in f
    ]

dataset = DamageDataset(
    test_files,
    "exp_alu_steel_1_ellip_dam/train_data",
    "exp_alu_steel_1_ellip_dam/train_masks"
)

# =====================================================
# Choose sample
# =====================================================

sample_idx = 0

signal, gt_mask = dataset[sample_idx]

input_tensor = (
    signal
    .unsqueeze(0)
    .to(DEVICE)
)

# =====================================================
# Prediction
# =====================================================

with torch.no_grad():

    pred = torch.sigmoid(
        model(input_tensor)
    )

pred_mask = (
    pred.squeeze()
    .cpu()
    .numpy()
)

binary_pred = (
    pred_mask > 0.5
).astype(np.float32)

# =====================================================
# Wrapper for segmentation explanation
# =====================================================

mask_tensor = torch.tensor(
    binary_pred,
    dtype=torch.float32,
    device=DEVICE
)

wrapped_model = SegmentationWrapper(
    model,
    mask_tensor
)

# =====================================================
# Integrated Gradients
# =====================================================

ig = IntegratedGradients(
    wrapped_model
)

attributions = ig.attribute(
    input_tensor,
    baselines=torch.zeros_like(input_tensor),
    n_steps=50
)

attributions = (
    attributions
    .squeeze()
    .detach()
    .cpu()
    .numpy()
)

# =====================================================
# Select frame for visualization
# =====================================================

frame_idx = 40

frame = signal[
    frame_idx
].numpy()

attr = attributions[
    frame_idx
]

# =====================================================
# Normalize images
# =====================================================

frame = (
    frame - frame.min()
) / (
    frame.max() - frame.min() + 1e-8
)

attr = np.abs(attr)

attr = (
    attr - attr.min()
) / (
    attr.max() - attr.min() + 1e-8
)

# =====================================================
# Plot
# =====================================================

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(
    frame,
    cmap="gray"
)
plt.title(
    f"Signal Frame {frame_idx}"
)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(
    gt_mask.squeeze(),
    cmap="gray"
)
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(pred_mask, cmap="jet")
plt.title("Prediction")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(
    frame,
    cmap="gray"
)

plt.imshow(
    attr,
    cmap="jet",
    alpha=0.5
)

plt.title("Integrated Gradients")
plt.axis("off")

plt.tight_layout()
plt.show()