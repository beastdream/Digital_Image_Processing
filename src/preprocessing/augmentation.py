import cv2
import numpy as np
import random
from typing import List, Tuple, Optional

class DataAugmenter:
    """
    Data Augmentation pipeline for Road Damage Detection.
    CRITICAL: Applied ONLY to Training split. Validation & Test splits are NEVER augmented.
    Synchronously updates Bounding Boxes for geometric transformations.
    """

    def __init__(self, config: dict):
        self.config = config

    def apply(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        split: str = "train"
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Applies configured augmentations if split == 'train' and config is enabled.
        """
        if split != "train" or not self.config.get("enabled", False):
            return image.copy(), [list(b) for b in (bboxes or [])]

        img_aug = image.copy()
        bbox_aug = [list(b) for b in (bboxes or [])]

        # 1. Horizontal Flip
        flip_cfg = self.config.get("horizontal_flip", {})
        if flip_cfg.get("prob", 0.0) > 0 and random.random() < flip_cfg["prob"]:
            img_aug, bbox_aug = self._horizontal_flip(img_aug, bbox_aug)

        # 2. Translation
        trans_cfg = self.config.get("translation", {})
        if trans_cfg.get("prob", 0.0) > 0 and random.random() < trans_cfg["prob"]:
            img_aug, bbox_aug = self._translation(img_aug, bbox_aug, trans_cfg.get("translate_range", [0.05, 0.05]))

        # 3. Scaling
        scale_cfg = self.config.get("scaling", {})
        if scale_cfg.get("prob", 0.0) > 0 and random.random() < scale_cfg["prob"]:
            img_aug, bbox_aug = self._scaling(img_aug, bbox_aug, scale_cfg.get("scale_range", [0.85, 1.15]))

        # 4. Rotation
        rot_cfg = self.config.get("rotation", {})
        if rot_cfg.get("prob", 0.0) > 0 and random.random() < rot_cfg["prob"]:
            img_aug, bbox_aug = self._rotation(img_aug, bbox_aug, rot_cfg.get("max_angle_deg", 10))

        # 5. Brightness & Contrast
        bc_cfg = self.config.get("brightness_contrast", {})
        if bc_cfg.get("prob", 0.0) > 0 and random.random() < bc_cfg["prob"]:
            img_aug = self._brightness_contrast(
                img_aug,
                bc_cfg.get("brightness_range", [0.8, 1.2]),
                bc_cfg.get("contrast_range", [0.8, 1.2])
            )

        # 6. Blur & Noise
        bn_cfg = self.config.get("blur_noise", {})
        if bn_cfg.get("prob", 0.0) > 0 and random.random() < bn_cfg["prob"]:
            img_aug = self._blur_noise(img_aug, bn_cfg.get("blur_kernel", 3), bn_cfg.get("noise_std", 5.0))

        return img_aug, bbox_aug

    def _horizontal_flip(
        self,
        image: np.ndarray,
        bboxes: List[List[float]]
    ) -> Tuple[np.ndarray, List[List[float]]]:
        flipped_img = cv2.flip(image, 1)
        flipped_bboxes = []
        for bbox in bboxes:
            cls_id, xc, yc, w, h = bbox
            xc_flip = 1.0 - xc
            flipped_bboxes.append([cls_id, xc_flip, yc, w, h])
        return flipped_img, flipped_bboxes

    def _translation(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        translate_range: List[float]
    ) -> Tuple[np.ndarray, List[List[float]]]:
        h_img, w_img = image.shape[:2]
        tx_max, ty_max = translate_range

        dx_norm = random.uniform(-tx_max, tx_max)
        dy_norm = random.uniform(-ty_max, ty_max)

        dx_px = dx_norm * w_img
        dy_px = dy_norm * h_img

        M = np.float32([[1, 0, dx_px], [0, 1, dy_px]])
        shifted_img = cv2.warpAffine(image, M, (w_img, h_img), borderValue=(114, 114, 114))

        shifted_bboxes = []
        for bbox in bboxes:
            cls_id, xc, yc, w, h = bbox
            xc_new = xc + dx_norm
            yc_new = yc + dy_norm

            xmin = max(0.0, min(1.0, xc_new - w / 2.0))
            ymin = max(0.0, min(1.0, yc_new - h / 2.0))
            xmax = max(0.0, min(1.0, xc_new + w / 2.0))
            ymax = max(0.0, min(1.0, yc_new + h / 2.0))

            w_new = xmax - xmin
            h_new = ymax - ymin
            xc_clip = (xmin + xmax) / 2.0
            yc_clip = (ymin + ymax) / 2.0

            if w_new > 0.01 and h_new > 0.01:
                shifted_bboxes.append([cls_id, xc_clip, yc_clip, w_new, h_new])

        return shifted_img, shifted_bboxes

    def _scaling(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        scale_range: List[float]
    ) -> Tuple[np.ndarray, List[List[float]]]:
        h_img, w_img = image.shape[:2]
        scale = random.uniform(scale_range[0], scale_range[1])

        cx, cy = w_img / 2.0, h_img / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        scaled_img = cv2.warpAffine(image, M, (w_img, h_img), borderValue=(114, 114, 114))

        scaled_bboxes = []
        for bbox in bboxes:
            cls_id, xc, yc, w, h = bbox

            xc_new = 0.5 + (xc - 0.5) * scale
            yc_new = 0.5 + (yc - 0.5) * scale
            w_new = w * scale
            h_new = h * scale

            xmin = max(0.0, min(1.0, xc_new - w_new / 2.0))
            ymin = max(0.0, min(1.0, yc_new - h_new / 2.0))
            xmax = max(0.0, min(1.0, xc_new + w_new / 2.0))
            ymax = max(0.0, min(1.0, yc_new + h_new / 2.0))

            w_clip = xmax - xmin
            h_clip = ymax - ymin
            xc_clip = (xmin + xmax) / 2.0
            yc_clip = (ymin + ymax) / 2.0

            if w_clip > 0.01 and h_clip > 0.01:
                scaled_bboxes.append([cls_id, xc_clip, yc_clip, w_clip, h_clip])

        return scaled_img, scaled_bboxes

    def _rotation(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        max_angle_deg: float
    ) -> Tuple[np.ndarray, List[List[float]]]:
        h_img, w_img = image.shape[:2]
        angle = random.uniform(-max_angle_deg, max_angle_deg)

        cx, cy = w_img / 2.0, h_img / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated_img = cv2.warpAffine(image, M, (w_img, h_img), borderValue=(114, 114, 114))

        rotated_bboxes = []
        for bbox in bboxes:
            cls_id, xc, yc, w, h = bbox

            # 4 bbox corners in normalized coords
            corners = np.array([
                [xc - w/2, yc - h/2],
                [xc + w/2, yc - h/2],
                [xc + w/2, yc + h/2],
                [xc - w/2, yc + h/2]
            ])

            # Convert to pixels
            corners_px = corners * np.array([w_img, h_img])
            ones = np.ones((4, 1))
            corners_homo = np.hstack([corners_px, ones])

            # Transform corners
            rot_corners_px = (M @ corners_homo.T).T

            # Normalize rotated corners
            rot_corners_norm = rot_corners_px / np.array([w_img, h_img])

            xmin = max(0.0, min(1.0, float(np.min(rot_corners_norm[:, 0]))))
            xmax = max(0.0, min(1.0, float(np.max(rot_corners_norm[:, 0]))))
            ymin = max(0.0, min(1.0, float(np.min(rot_corners_norm[:, 1]))))
            ymax = max(0.0, min(1.0, float(np.max(rot_corners_norm[:, 1]))))

            w_new = xmax - xmin
            h_new = ymax - ymin
            xc_new = (xmin + xmax) / 2.0
            yc_new = (ymin + ymax) / 2.0

            if w_new > 0.01 and h_new > 0.01:
                rotated_bboxes.append([cls_id, xc_new, yc_new, w_new, h_new])

        return rotated_img, rotated_bboxes

    def _brightness_contrast(
        self,
        image: np.ndarray,
        brightness_range: List[float],
        contrast_range: List[float]
    ) -> np.ndarray:
        alpha = random.uniform(contrast_range[0], contrast_range[1])
        beta = (random.uniform(brightness_range[0], brightness_range[1]) - 1.0) * 255.0
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def _blur_noise(
        self,
        image: np.ndarray,
        blur_kernel: int,
        noise_std: float
    ) -> np.ndarray:
        out = image.copy()
        if random.random() < 0.5:
            k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            out = cv2.GaussianBlur(out, (k, k), 0)
        if random.random() < 0.5:
            noise = np.random.normal(0, noise_std, out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out
