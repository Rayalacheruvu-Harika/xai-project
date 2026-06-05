from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sample = sorted(
    Path("exp_alu_steel_1_ellip_dam/train_data").glob("*.npy")
)[0]

signal = np.load(sample)

print(signal.shape)

frames = [0, 20, 40, 60, 82]

fig, axes = plt.subplots(1, 5, figsize=(18,4))

for ax, idx in zip(axes, frames):
    ax.imshow(signal[idx])
    ax.set_title(f"Frame {idx}")
    ax.axis("off")

plt.tight_layout()
plt.show()