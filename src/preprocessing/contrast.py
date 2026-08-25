import cv2
import numpy as np
from typing import Tuple

def apply_global_contrast(
    image: np.ndarray,
    factor: float = 1.2
) -> np.ndarray:
    """
    Applies global linear contrast scaling: I_out = clip(factor * I_in, 0, 255).
    """
    adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return adjusted

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel
    in LAB color space to enhance local contrast of road texture without color distortion.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return enhanced_bgr
