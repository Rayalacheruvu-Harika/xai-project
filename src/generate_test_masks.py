import re
import cv2
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("generated_test_masks")
OUTPUT_DIR.mkdir(exist_ok=True)

with open("splits/test.txt") as f:
    test_files = [
        line.strip()
        for line in f.readlines()
    ]

pattern = r'xy_(\d+)_(\d+)_ab_(\d+)_(\d+)_th_(\d+)'

for filename in test_files:

    m = re.search(pattern, filename)

    cx = int(m.group(1))
    cy = int(m.group(2))
    a = int(m.group(3))
    b = int(m.group(4))
    theta = float(m.group(5))

    mask = np.zeros((467, 467), dtype=np.uint8)

    cv2.ellipse(
        mask,
        (cx, cy),
        (a, b),
        theta,
        0,
        360,
        255,
        -1
    )

    base_name = (
        filename
        .replace("numpy-signal-Vx-", "")
        .replace(".npy", "")
    )

    cv2.imwrite(
        str(OUTPUT_DIR / f"{base_name}_mask.png"),
        mask
    )

print("Generated", len(test_files), "test masks")