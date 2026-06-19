import torch
import numpy as np

from pathlib import Path
from PIL import Image

from pytorch_grad_cam import GradCAM

from dataset import DamageDataset
from model import CustomUNet


class SegmentationTarget:

    def __init__(self, mask):
        self.mask = torch.tensor(mask)

    def __call__(self, model_output):

        if self.mask.device != model_output.device:
            self.mask = self.mask.to(model_output.device)

        return (
            model_output.squeeze() *
            self.mask
        ).sum()


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SAVE_DIR = Path("gradcam_maps")
SAVE_DIR.mkdir(exist_ok=True)

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

target_layer = model.bottleneck

cam = GradCAM(
    model=model,
    target_layers=[target_layer]
)

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

for idx, filename in enumerate(test_files):

    signal, _ = dataset[idx]

    input_tensor = (
        signal
        .unsqueeze(0)
        .to(DEVICE)
    )

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

    targets = [
        SegmentationTarget(
            binary_pred
        )
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    grayscale_cam = (
        grayscale_cam -
        grayscale_cam.min()
    ) / (
        grayscale_cam.max() -
        grayscale_cam.min() +
        1e-8
    )

    grayscale_cam = (
        grayscale_cam * 255
    ).astype(np.uint8)

    Image.fromarray(
        grayscale_cam
    ).save(

        SAVE_DIR /
        f"{Path(filename).stem}_gradcam.png"
    )

print("GradCAM maps saved")