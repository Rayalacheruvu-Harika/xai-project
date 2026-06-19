import torch
import numpy as np
import pandas as pd

from dataset import DamageDataset
from model import CustomUNet

from pytorch_grad_cam import GradCAM
from captum.attr import IntegratedGradients


DEVICE = "cpu"


# --------------------------------------------------
# Segmentation target
# --------------------------------------------------

class SegmentationTarget:

    def __init__(self, mask):
        self.mask = torch.tensor(mask)

    def __call__(self, model_output):

        if self.mask.device != model_output.device:
            self.mask = self.mask.to(
                model_output.device
            )

        return (
            model_output.squeeze() *
            self.mask
        ).sum()


# --------------------------------------------------
# IG wrapper
# --------------------------------------------------

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


# --------------------------------------------------
# Alignment metric
# --------------------------------------------------

def alignment_score(attr, gt):

    attr = np.maximum(attr, 0)

    return (
        attr[gt > 0].sum()
    ) / (
        attr.sum() + 1e-8
    )


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = CustomUNet(
    in_channels=83,
    out_channels=1
)

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

# --------------------------------------------------
# Dataset
# --------------------------------------------------

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

noise_levels = [
    0.0,
    0.01,
    0.05
]

results = []

# --------------------------------------------------
# Loop
# --------------------------------------------------

for noise_std in noise_levels:

    gradcam_scores = []
    ig_scores = []

    for idx in range(len(dataset)):

        signal, gt_mask = dataset[idx]

        signal_np = signal.numpy()

        noise = np.random.normal(
            0,
            noise_std,
            signal_np.shape
        )

        signal_np = (
            signal_np + noise
        )

        input_tensor = torch.tensor(
            signal_np,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            pred = torch.sigmoid(
                model(input_tensor)
            )

        pred_mask = (
            pred.squeeze()
            .numpy()
        )

        binary_pred = (
            pred_mask > 0.5
        ).astype(np.float32)

        # ------------------
        # GradCAM
        # ------------------

        targets = [
            SegmentationTarget(
                binary_pred
            )
        ]

        gradcam_map = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

        gradcam_scores.append(
            alignment_score(
                gradcam_map,
                gt_mask.squeeze().numpy()
            )
        )

        # ------------------
        # IG
        # ------------------

        mask_tensor = torch.tensor(
            binary_pred,
            dtype=torch.float32
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

        attr = ig.attribute(
            input_tensor,
            baselines=torch.zeros_like(
                input_tensor
            ),
            n_steps=20
        )

        attr = (
            attr.squeeze()
            .detach()
            .numpy()
        )

        attr = np.abs(
            attr[40]
        )

        ig_scores.append(
            alignment_score(
                attr,
                gt_mask.squeeze().numpy()
            )
        )

    results.append([
        noise_std,
        np.mean(gradcam_scores),
        np.mean(ig_scores)
    ])

df = pd.DataFrame(
    results,
    columns=[
        "noise",
        "gradcam_alignment",
        "ig_alignment"
    ]
)

print(df)

df.to_csv(
    "results/noise_xai.csv",
    index=False
)