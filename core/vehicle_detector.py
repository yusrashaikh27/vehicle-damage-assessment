"""Vehicle segmentation for scale-invariant damage assessment and false positive reduction.

This module provides vehicle segmentation using a COCO-pretrained model to:
1. Create a vehicle mask for scale-invariant severity assessment
2. Filter out false positive detections on background/road
3. Enable more accurate area fraction calculation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ultralytics import YOLO

# Vehicle classes in COCO dataset that we consider part of the vehicle
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck


class VehicleDetector:
    """Detects and segments vehicles in images using COCO-pretrained model."""

    def __init__(self, weights_path: str | Path = "yolov8n-seg.pt"):
        """
        Initialize vehicle detector.

        Args:
            weights_path: Path to YOLOv8 segmentation weights.
                         Defaults to yolov8n-seg.pt (will download if not present).
        """
        self.model = YOLO(weights_path)
        self.weights_path = str(weights_path)

    def detect_vehicle(self, image_path: str | Path) -> tuple[Optional[np.ndarray], float]:
        """
        Detect vehicle in image and return segmentation mask and coverage ratio.

        Args:
            image_path: Path to input image

        Returns:
            tuple: (vehicle_mask, coverage_ratio)
                - vehicle_mask: Boolean numpy array (H, W) where True indicates vehicle pixels
                - coverage_ratio: Fraction of image covered by vehicle mask (0-1)
                - Returns (None, 0.0) if no vehicle detected with sufficient confidence
        """
        # Run inference
        results = self.model(str(image_path), verbose=False)[0]

        # Initialize empty mask
        if results.orig_img is None:
            return None, 0.0

        h, w = results.orig_img.shape[:2]
        vehicle_mask = np.zeros((h, w), dtype=bool)
        vehicle_pixels = 0

        # Process segmentation results
        if results.masks is not None and results.boxes is not None:
            masks = results.masks.data.cpu().numpy()  # (n, h, w)
            boxes = results.boxes.data.cpu().numpy()  # (n, 6) - x1, y1, x2, y2, conf, cls
            classes = boxes[:, 5].astype(int)
            confidences = boxes[:, 4]

            # Process each detection
            for mask, cls, conf in zip(masks, classes, confidences):
                # Check if this is a vehicle class and has sufficient confidence
                if cls in VEHICLE_CLASSES and conf > 0.5:
                    # Resize mask to match image dimensions if needed
                    if mask.shape != (h, w):
                        import cv2
                        mask = cv2.resize(mask.astype(np.float32), (w, h),
                                        interpolation=cv2.INTER_LINEAR) > 0.5

                    vehicle_mask |= mask.astype(bool)
                    vehicle_pixels += np.sum(mask)

        # Calculate coverage ratio
        total_pixels = h * w
        coverage_ratio = vehicle_pixels / total_pixels if total_pixels > 0 else 0.0

        # Return mask only if vehicle covers reasonable portion of image
        # ( prevents using tiny/distant vehicle detections as reference )
        if coverage_ratio > 0.05:  # At least 5% of image
            return vehicle_mask, coverage_ratio
        else:
            return None, 0.0

    def get_vehicle_area(self, image_path: str | Path) -> Optional[float]:
        """
        Get the vehicle mask area in pixels.

        Args:
            image_path: Path to input image

        Returns:
            Vehicle area in pixels, or None if no reliable vehicle detected
        """
        mask, _ = self.detect_vehicle(image_path)
        if mask is not None:
            return float(np.sum(mask))
        return None


# Convenience function for backward compatibility
def estimate_vehicle_area(image_path: str | Path) -> Optional[float]:
    """
    Estimate vehicle area in pixels using COCO-pretrained segmentation model.

    Args:
        image_path: Path to input image

    Returns:
        Vehicle area in pixels, or None if detection fails
    """
    detector = VehicleDetector()
    return detector.get_vehicle_area(image_path)