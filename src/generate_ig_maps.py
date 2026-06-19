import torch
import numpy as np

from pathlib import Path
from PIL import Image

from captum.attr import IntegratedGradients

from dataset import DamageDataset
from model import CustomUNet


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


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SAVE_DIR = Path("ig_maps")
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

    mask_tensor = torch.tensor(
        binary_pred,
        dtype=torch.float32,
        device=DEVICE
    )

    wrapped_model = (
        SegmentationWrapper(
            model,
            mask_tensor
        )
    )

    ig = IntegratedGradients(
        wrapped_model
    )

    attributions = ig.attribute(
        input_tensor,
        baselines=torch.zeros_like(
            input_tensor
        ),
        n_steps=50
    )

    attributions = (
        attributions
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    attr = np.abs(
        attributions[40]
    )

    attr = (
        attr - attr.min()
    ) / (
        attr.max() -
        attr.min() +
        1e-8
    )

    attr = (
        attr * 255
    ).astype(np.uint8)

    Image.fromarray(
        attr
    ).save(

        SAVE_DIR /
        f"{Path(filename).stem}_ig.png"
    )

print("IG maps saved")