import cv2
import numpy as np
from typing import Tuple

def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (3, 3),
    sigma: float = 0.5
) -> np.ndarray:
    """
    Applies Gaussian Blur smoothing filter to reduce high-frequency image noise.
    """
    # Ensure kernel size values are odd positive integers
    k_w = kernel_size[0] if kernel_size[0] % 2 == 1 else kernel_size[0] + 1
    k_h = kernel_size[1] if kernel_size[1] % 2 == 1 else kernel_size[1] + 1
    return cv2.GaussianBlur(image, (k_w, k_h), sigmaX=sigma, sigmaY=sigma)

def apply_median_blur(
    image: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Applies Median Filter blur to remove salt-and-pepper noise while retaining edge sharpness.
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.medianBlur(image, k)
