from pathlib import Path
import numpy as np


def compute_mask_centroid(mask: np.ndarray):
    """
    mask: [H, W] binary numpy array
    returns (cx, cy) as integers

    If mask is empty, return image center.
    """
    ys, xs = np.where(mask > 0)

    h, w = mask.shape

    if len(xs) == 0:
        return w // 2, h // 2

    cx = int(round(xs.mean()))
    cy = int(round(ys.mean()))
    return cx, cy


def compute_pred_centroid_from_prob(prob_map: np.ndarray, threshold: float = 0.5):
    """
    prob_map: [H, W] float in [0,1]

    Strategy:
    1. threshold
    2. if foreground exists -> centroid of foreground
    3. else -> argmax location
    """
    binary = prob_map > threshold
    ys, xs = np.where(binary)

    if len(xs) > 0:
        cx = int(round(xs.mean()))
        cy = int(round(ys.mean()))
        return cx, cy

    # fallback: argmax
    flat_idx = np.argmax(prob_map)
    h, w = prob_map.shape
    cy, cx = np.unravel_index(flat_idx, (h, w))
    return int(cx), int(cy)


def crop_with_padding(image: np.ndarray, center_x: int, center_y: int, crop_size: int):
    """
    image:
        [C, H, W] or [H, W]

    returns:
        crop, meta

    crop is always crop_size x crop_size using zero padding if needed.

    meta contains information needed to paste back later.
    """
    if image.ndim == 3:
        c, h, w = image.shape
        is_channel_first = True
    elif image.ndim == 2:
        h, w = image.shape
        is_channel_first = False
    else:
        raise ValueError("image must be [C,H,W] or [H,W]")

    half = crop_size // 2

    x1 = center_x - half
    y1 = center_y - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if is_channel_first:
        crop = np.zeros((c, crop_size, crop_size), dtype=image.dtype)
        crop[:, dst_y1:dst_y2, dst_x1:dst_x2] = image[:, src_y1:src_y2, src_x1:src_x2]
    else:
        crop = np.zeros((crop_size, crop_size), dtype=image.dtype)
        crop[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    meta = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "src_x1": src_x1,
        "src_y1": src_y1,
        "src_x2": src_x2,
        "src_y2": src_y2,
        "dst_x1": dst_x1,
        "dst_y1": dst_y1,
        "dst_x2": dst_x2,
        "dst_y2": dst_y2,
        "crop_size": crop_size,
        "full_h": h,
        "full_w": w,
    }

    return crop, meta


def paste_crop_back(crop_pred: np.ndarray, meta: dict):
    """
    crop_pred: [crop_size, crop_size] prediction from Stage 2
    returns full-size [H, W] image with crop pasted back
    """
    full_h = meta["full_h"]
    full_w = meta["full_w"]

    canvas = np.zeros((full_h, full_w), dtype=crop_pred.dtype)

    src = crop_pred[
        meta["dst_y1"]:meta["dst_y2"],
        meta["dst_x1"]:meta["dst_x2"]
    ]

    canvas[
        meta["src_y1"]:meta["src_y2"],
        meta["src_x1"]:meta["src_x2"]
    ] = src

    return canvas