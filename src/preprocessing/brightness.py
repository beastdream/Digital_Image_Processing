import cv2
import numpy as np

def adjust_brightness(
    image: np.ndarray,
    brightness_factor: float = 1.0
) -> np.ndarray:
    """
    Adjusts image brightness using smooth HSV Value channel scaling to avoid saturation clipping artifacts.

    Args:
        image: Input BGR image.
        brightness_factor: Scale factor (> 1.0 brightens, < 1.0 darkens, 1.0 keeps unchanged).

    Returns:
        Adjusted BGR image.
    """
    if abs(brightness_factor - 1.0) < 1e-4:
        return image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
    adjusted_hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
