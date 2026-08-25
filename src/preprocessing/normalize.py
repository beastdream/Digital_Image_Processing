import numpy as np
from typing import Optional, List, Tuple

def normalize_image(
    image: np.ndarray,
    mode: str = "pixel_scale",
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> np.ndarray:
    """
    Normalizes image pixel values for model ingestion.

    Note on Normalization Responsibilities:
    - Image Preprocessing Normalization converts uint8 [0, 255] images to float32 [0.0, 1.0] or standardized values.
    - YOLO Internal Preprocessing (Ultralytics YOLO) expects uint8 BGR/RGB images directly and performs internal
      normalization on GPU. For YOLO dataset saving, images should remain uint8 [0, 255], while float32 arrays
      are used for direct PyTorch PyTorch model inference pipelines.

    Args:
        image: Input uint8 BGR image (H, W, C).
        mode: "pixel_scale" ([0.0, 1.0]), "standardize" (mean/std), or "none".
        mean: List of RGB mean values for standardization.
        std: List of RGB std values for standardization.

    Returns:
        Normalized image float32 or uint8 array.
    """
    if mode == "none" or mode is None:
        return image

    img_float = image.astype(np.float32) / 255.0

    if mode == "pixel_scale":
        return img_float

    elif mode == "standardize":
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        # Convert BGR float to RGB float for standardization
        img_rgb = cv2_bgr_to_rgb(img_float)
        mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)

        standardized = (img_rgb - mean_arr) / std_arr
        return standardized

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

def cv2_bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return img[:, :, ::-1]
