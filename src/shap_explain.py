
import torch
import torch.nn as nn
import shap
import numpy as np
from pathlib import Path

from dataset import DamageDataset
from model import CustomUNet


DEVICE = "cpu"

DATA_DIR = "exp_alu_steel_1_ellip_dam/train_data"
MASK_DIR = "exp_alu_steel_1_ellip_dam/train_masks"
MODEL_PATH = "checkpoints/best_model.pth"

with open("splits/test.txt") as f:
    test_files = [x.strip() for x in f]


class SegmentationWrapper(nn.Module):
    """
    Wrap the segmentation network so SHAP receives
    one scalar output per image.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        probs = torch.sigmoid(logits)

        # Mean probability over the output mask.
        # This converts [B,1,H,W] -> [B]
        score = probs.mean(dim=(1,2,3))

        return score.unsqueeze(1)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = DamageDataset(
    test_files,
    DATA_DIR,
    MASK_DIR
)

# --------------------------------------------------
# Model
# --------------------------------------------------

model = CustomUNet(
    in_channels=83,
    out_channels=1
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

wrapped_model = SegmentationWrapper(model)

# --------------------------------------------------
# Background samples
# --------------------------------------------------

background = []

n_background = min(5, len(dataset))

for i in range(n_background):
    signal, _ = dataset[i]
    background.append(signal)

background = torch.stack(background).to(DEVICE)

# --------------------------------------------------
# SHAP Explainer
# --------------------------------------------------

explainer = shap.GradientExplainer(
    wrapped_model,
    background
)

# --------------------------------------------------
# Explain every test image
# --------------------------------------------------

output_dir = Path("shap_results")
output_dir.mkdir(exist_ok=True)

for idx, filename in enumerate(test_files):

    signal, _ = dataset[idx]

    signal = signal.unsqueeze(0).to(DEVICE)

    print(f"Explaining {filename}")

    shap_values = explainer.shap_values(signal)
    print(type(shap_values))

    if isinstance(shap_values, list):
        print("Length:", len(shap_values))
        print("Shape:", np.array(shap_values[0]).shape)
    else:
        print("Shape:", np.array(shap_values).shape)

    np.save(
        output_dir / f"{Path(filename).stem}_shap.npy",
        np.asarray(shap_values)
    )

print("\\nFinished.")
print("SHAP values saved in:", output_dir)
