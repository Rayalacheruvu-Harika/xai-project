import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import DamageDataset
from model import CustomUNet


# =====================================================
# Custom target for segmentation
# =====================================================

class SegmentationTarget:

    def __init__(self, mask):
        self.mask = torch.tensor(mask)

    def __call__(self, model_output):

        if self.mask.device != model_output.device:
            self.mask = self.mask.to(model_output.device)

        return (model_output.squeeze() * self.mask).sum()


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
# Generate prediction
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
# Grad-CAM
# =====================================================

target_layer = model.bottleneck

cam = GradCAM(
    model=model,
    target_layers=[target_layer]
)

targets = [
    SegmentationTarget(binary_pred)
]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]

# =====================================================
# Visualization frame
# =====================================================

frame_idx = 40

frame = signal[frame_idx].numpy()

frame = (
    frame - frame.min()
) / (
    frame.max() - frame.min() + 1e-8
)

rgb_frame = np.stack(
    [frame, frame, frame],
    axis=-1
)

cam_image = show_cam_on_image(
    rgb_frame,
    grayscale_cam,
    use_rgb=True
)

# =====================================================
# Plot
# =====================================================

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(frame, cmap="gray")
plt.title(f"Signal Frame {frame_idx}")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(
    gt_mask.squeeze(),
    cmap="gray"
)
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(
    binary_pred,
    cmap="gray"
)
plt.title("Prediction")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(cam_image)
plt.title("Grad-CAM")
plt.axis("off")

plt.tight_layout()
plt.show()