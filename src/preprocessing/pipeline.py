import os
import yaml
import numpy as np
from typing import Tuple, List, Dict, Union, Optional

from src.preprocessing.resize import direct_resize, letterbox_resize
from src.preprocessing.denoise import apply_gaussian_blur, apply_median_blur
from src.preprocessing.contrast import apply_global_contrast, apply_clahe
from src.preprocessing.brightness import adjust_brightness
from src.preprocessing.normalize import normalize_image
from src.preprocessing.augmentation import DataAugmenter

class ImagePreprocessingPipeline:
    """
    Unified Image Preprocessing Pipeline for Road Damage Detection.
    Encapsulates sequential stages:
    Raw Image -> [Augmentation (Train only)] -> Resize -> Denoise -> Contrast -> Brightness -> Normalization
    """

    def __init__(self, config: Union[dict, str]):
        if isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

        # Dataset images written for Ultralytics must remain uint8. Normalization is
        # only useful for direct array consumers, never for persisted YOLO images.
        self.config = self.config or {}

        self.augmenter = DataAugmenter(self.config.get("augmentation", {}))

    def process(
        self,
        image: np.ndarray,
        bboxes: Optional[List[List[float]]] = None,
        split: str = "train",
        return_intermediates: bool = False
    ) -> Tuple[np.ndarray, Optional[List[List[float]]], Dict]:
        """
        Executes preprocessing steps sequentially based on configuration.

        Args:
            image: Input uint8 BGR image (H, W, C).
            bboxes: Optional list of normalized YOLO bboxes [[cls, xc, yc, w, h], ...].
            split: Dataset split name ("train", "val", "test").
            return_intermediates: If True, returns dictionary of step-by-step intermediate images.

        Returns:
            final_image: Preprocessed image array.
            final_bboxes: Updated normalized bboxes.
            metadata: Execution info and intermediate images if requested.
        """
        curr_img = image.copy()
        curr_bboxes = [list(b) for b in bboxes] if bboxes is not None else None

        intermediates = {"original": curr_img.copy()}
        bbox_intermediates = {"original": [list(box) for box in curr_bboxes] if curr_bboxes is not None else None}
        exec_meta = {"split": split, "steps_executed": []}

        # 1. Augmentation (Train split only)
        aug_cfg = self.config.get("augmentation", {})
        if split == "train" and aug_cfg.get("enabled", False):
            curr_img, curr_bboxes = self.augmenter.apply(curr_img, curr_bboxes, split="train")
            exec_meta["steps_executed"].append("augmentation")
            if return_intermediates:
                intermediates["augmentation"] = curr_img.copy()
                bbox_intermediates["augmentation"] = [list(box) for box in curr_bboxes]

        # 2. Resize
        resize_cfg = self.config.get("resize", {})
        if resize_cfg.get("enabled", True):
            mode = resize_cfg.get("mode", "letterbox")
            target_size = tuple(resize_cfg.get("target_size", [640, 640]))
            pad_color = tuple(resize_cfg.get("pad_color", [114, 114, 114]))

            if mode == "letterbox":
                curr_img, curr_bboxes, pad_meta = letterbox_resize(
                    curr_img, target_size=target_size, bboxes=curr_bboxes, pad_color=pad_color
                )
                exec_meta["pad_meta"] = pad_meta
            elif mode == "direct":
                curr_img = direct_resize(curr_img, target_size=target_size)
                # Direct resize retains normalized bboxes
            else:
                raise ValueError(f"Unsupported resize mode: {mode}")

            exec_meta["steps_executed"].append(f"resize_{mode}")
            if return_intermediates:
                intermediates["resize"] = curr_img.copy()
                bbox_intermediates["resize"] = [list(box) for box in curr_bboxes] if curr_bboxes is not None else None

        # 3. Denoising
        denoise_cfg = self.config.get("denoise", {})
        if denoise_cfg.get("enabled", False):
            method = denoise_cfg.get("method", "gaussian")
            if method == "gaussian":
                k_size = tuple(denoise_cfg.get("gaussian_kernel", [3, 3]))
                sigma = float(denoise_cfg.get("gaussian_sigma", 0.5))
                curr_img = apply_gaussian_blur(curr_img, kernel_size=k_size, sigma=sigma)
            elif method == "median":
                k_size = int(denoise_cfg.get("median_kernel", 3))
                curr_img = apply_median_blur(curr_img, kernel_size=k_size)
            else:
                raise ValueError(f"Unsupported denoise method: {method}")

            exec_meta["steps_executed"].append(f"denoise_{method}")
            if return_intermediates:
                intermediates["denoise"] = curr_img.copy()

        # 4. Contrast Enhancement
        contrast_cfg = self.config.get("contrast", {})
        if contrast_cfg.get("enabled", False):
            method = contrast_cfg.get("method", "clahe")
            if method == "clahe":
                clip_limit = float(contrast_cfg.get("clahe_clip_limit", 2.0))
                tile_size = tuple(contrast_cfg.get("clahe_tile_grid_size", [8, 8]))
                curr_img = apply_clahe(curr_img, clip_limit=clip_limit, tile_grid_size=tile_size)
            elif method == "global":
                factor = float(contrast_cfg.get("global_factor", 1.2))
                curr_img = apply_global_contrast(curr_img, factor=factor)
            else:
                raise ValueError(f"Unsupported contrast method: {method}")

            exec_meta["steps_executed"].append(f"contrast_{method}")
            if return_intermediates:
                intermediates["contrast"] = curr_img.copy()

        # 5. Brightness Adjustment
        bright_cfg = self.config.get("brightness", {})
        if bright_cfg.get("enabled", False):
            factor = float(bright_cfg.get("brightness_factor", 1.0))
            curr_img = adjust_brightness(curr_img, brightness_factor=factor)

            exec_meta["steps_executed"].append("brightness")
            if return_intermediates:
                intermediates["brightness"] = curr_img.copy()

        # Save uint8 processed image for visual inspection / dataset saving
        if return_intermediates:
            intermediates["final_uint8"] = curr_img.copy()
            bbox_intermediates["final_uint8"] = [list(box) for box in curr_bboxes] if curr_bboxes is not None else None

        # 6. Normalization
        norm_cfg = self.config.get("normalization", {})
        if norm_cfg.get("enabled", True):
            mode = norm_cfg.get("mode", "pixel_scale")
            mean = norm_cfg.get("mean", None)
            std = norm_cfg.get("std", None)
            final_img = normalize_image(curr_img, mode=mode, mean=mean, std=std)
            exec_meta["steps_executed"].append(f"normalize_{mode}")
        else:
            final_img = curr_img

        if return_intermediates:
            exec_meta["intermediates"] = intermediates
            exec_meta["bbox_intermediates"] = bbox_intermediates

        return final_img, curr_bboxes, exec_meta
