import cv2
import numpy as np
from typing import Tuple, List, Optional

def direct_resize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640)
) -> np.ndarray:
    """
    Directly resizes image to target_size (width, height) without aspect ratio preservation.
    Note: Normalized YOLO bboxes remain unchanged under direct scaling.
    """
    target_w, target_h = target_size
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

def letterbox_resize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    bboxes: Optional[List[List[float]]] = None,
    pad_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, Optional[List[List[float]]], dict]:
    """
    Resizes image to fit inside target_size (width, height) while preserving aspect ratio,
    padding remaining borders with pad_color. Synchronously updates normalized bboxes.

    Args:
        image: BGR image numpy array.
        target_size: (width, height) tuple.
        bboxes: List of [cls_id, xc, yc, w, h] normalized in [0, 1].
        pad_color: (B, G, R) tuple for border padding.

    Returns:
        padded_image: Resized & padded image array.
        updated_bboxes: List of updated [cls_id, xc, yc, w, h] bboxes (or None if input bboxes is None).
        meta: Dictionary with scaling and padding metadata.
    """
    h_orig, w_orig = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scale factor
    r = min(target_w / w_orig, target_h / h_orig)

    # Unpadded dimensions
    w_new = int(round(w_orig * r))
    h_new = int(round(h_orig * r))

    # Calculate padding (symmetric)
    dw = (target_w - w_new) / 2.0
    dh = (target_h - h_new) / 2.0

    # Resize image preserving aspect ratio
    if (w_orig, h_orig) != (w_new, h_new):
        resized_img = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    else:
        resized_img = image.copy()

    # Create padded background
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded_image = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=pad_color
    )

    # Ensure output matches target_size exactly
    if padded_image.shape[:2] != (target_h, target_w):
        padded_image = cv2.resize(padded_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    meta = {
        "scale_ratio": r,
        "pad_w": dw,
        "pad_h": dh,
        "orig_size": (w_orig, h_orig),
        "target_size": (target_w, target_h)
    }

    if bboxes is None:
        return padded_image, None, meta

    # Transform Bounding Boxes
    updated_bboxes = []
    for bbox in bboxes:
        cls_id, xc, yc, w, h = bbox

        # Convert normalized center to target padded image coordinates
        xc_new = (xc * w_orig * r + dw) / target_w
        yc_new = (yc * h_orig * r + dh) / target_h
        w_new_norm = (w * w_orig * r) / target_w
        h_new_norm = (h * h_orig * r) / target_h

        # Clip bounds safely to [0, 1]
        xmin = max(0.0, min(1.0, xc_new - w_new_norm / 2.0))
        ymin = max(0.0, min(1.0, yc_new - h_new_norm / 2.0))
        xmax = max(0.0, min(1.0, xc_new + w_new_norm / 2.0))
        ymax = max(0.0, min(1.0, yc_new + h_new_norm / 2.0))

        xc_clip = (xmin + xmax) / 2.0
        yc_clip = (ymin + ymax) / 2.0
        w_clip = xmax - xmin
        h_clip = ymax - ymin

        if w_clip > 0.0 and h_clip > 0.0:
            updated_bboxes.append([int(cls_id), xc_clip, yc_clip, w_clip, h_clip])

    return padded_image, updated_bboxes, meta
