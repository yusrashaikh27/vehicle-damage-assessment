"""YOLO wrapper with optional vehicle segmentation for scale-invariant assessment and false positive reduction.

This module extends the base YOLO detector to optionally include vehicle segmentation
capabilities for:
1. Creating vehicle masks for scale-invariant severity assessment
2. Filtering out false positive detections on background/road
3. Enabling more accurate area fraction calculation

Coordinate-space rule
---------------------
Everything this module returns - bbox, polygon, area_px, vehicle_mask,
vehicle_area_px - is in ORIGINAL PHOTO pixels. Ultralytics returns masks at the
model's inference size by default (e.g. 416x640) while boxes are scaled back to
the photo (e.g. 504x776); mixing the two caused the "operands could not be
broadcast together" failure. `retina_masks=True` plus the `_fit` guard below
keep every array in photo space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ultralytics import YOLO

from . import damage_config
from .vehicle_detector import VehicleDetector

Point = tuple[float, float]
BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel coordinates)


@dataclass(frozen=True)
class Detection:
    """The output of the detector for one damage instance."""
    class_name: str          # e.g. "scratch"
    bbox: BBox               # bounding box (x1, y1, x2, y2) in pixels
    polygon: List[Point]     # list of vertices (x, y) in pixels
    area_px: float           # area in pixels (from rasterized polygon mask)
    confidence: float        # detection confidence (0-1)
    # Kept for backward compatibility. Left as None per detection on purpose:
    # the single vehicle mask lives on DetectionResult, and copying a full-size
    # array onto every detection wasted memory.
    vehicle_mask: Optional[np.ndarray] = None


@dataclass(frozen=True)
class DetectionResult:
    """Result of running detection on one image."""
    detections: List[Detection]
    image_path: Path
    annotated_path: Optional[Path] = None
    vehicle_area_px: float = 0.0          # area of vehicle in pixels (0 if not detected)
    vehicle_mask: Optional[np.ndarray] = None  # vehicle segmentation mask
    image_width: int = 0
    image_height: int = 0
    is_stub: bool = False                 # True if using stub detector
    model_name: str = ""
    elapsed_ms: int = 0

    @property
    def damage_found(self) -> bool:
        return bool(self.detections)


def _fit(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return `mask` as a boolean array of exactly (height, width).

    A no-op when the size already matches. Nearest-neighbour keeps a binary mask
    binary; any interpolation would create in-between values. Note that
    cv2.resize takes (width, height), the reverse of numpy's .shape.
    """
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(
            (mask > 0.5).astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask > 0.5 if mask.dtype != bool else mask


class YoloSegDetector:
    """Wrapper around a YOLO instance-segmentation model with optional vehicle detection."""

    def __init__(
        self,
        weights: str | Path,
        vehicle_weights: str | Path = "yolov8n-seg.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        enable_vehicle_detection: bool = True,
    ):
        """
        Args:
            weights: path to the damage detection *.pt file
            vehicle_weights: path to vehicle segmentation *.pt file (COCO-pretrained)
            conf: detection confidence threshold
            iou: NMS IoU threshold
            imgsz: inference image size (pixels)
            enable_vehicle_detection: whether to run vehicle segmentation for masking
        """
        self.damage_model = YOLO(weights)
        self.vehicle_detector = (
            VehicleDetector(vehicle_weights) if enable_vehicle_detection else None
        )
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.enable_vehicle_detection = enable_vehicle_detection
        self.weights = Path(weights)
        self.vehicle_weights = Path(vehicle_weights) if enable_vehicle_detection else None
        self.model_name = (
            "YOLO11s-seg + "
            f"{'YOLOv8n-seg vehicle' if enable_vehicle_detection else 'no vehicle'}"
        )

    def detect(
        self,
        source: str | Path,
        *,
        augment: bool = False,
        annotate: bool = True,
        annotated_dir: Optional[Path | str] = None,
    ) -> DetectionResult:
        """Run damage detection on an image, optionally filtered by a vehicle mask.

        Returns a DetectionResult with detections in original-photo pixels, the
        vehicle mask/area (if enabled) and the annotated overlay path (if
        `annotate` is true and `annotated_dir` is given).
        """
        # retina_masks=True makes Ultralytics return masks at the ORIGINAL image
        # size instead of the inference size, so masks line up with the boxes.
        damage_results = self.damage_model(
            str(source),
            augment=augment,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            retina_masks=True,
            verbose=False,
        )[0]

        img_h, img_w = damage_results.orig_shape  # (height, width) of the photo

        # Vehicle segmentation, forced into photo space whatever size it returns.
        vehicle_mask: Optional[np.ndarray] = None
        vehicle_area_px = 0.0
        if self.enable_vehicle_detection and self.vehicle_detector is not None:
            raw_mask, _coverage = self.vehicle_detector.detect_vehicle(source)
            if raw_mask is not None:
                vehicle_mask = _fit(np.asarray(raw_mask), img_h, img_w)
                vehicle_area_px = float(vehicle_mask.sum())

        detections: List[Detection] = []
        if damage_results.boxes is not None and damage_results.masks is not None:
            boxes = damage_results.boxes.data.cpu().numpy()   # (n, 6)
            masks = damage_results.masks.data.cpu().numpy()   # (n, h, w)

            for box, raw in zip(boxes, masks):
                x1, y1, x2, y2, conf, cls = box
                class_id = int(cls)

                # Belt and braces: guarantee the mask is photo-sized and boolean
                # even if retina_masks behaves differently in another version.
                mask = _fit(raw, img_h, img_w)

                damage_area = float(mask.sum())
                if damage_area <= 0:
                    continue

                # Drop detections that sit mostly off the vehicle (road,
                # background). Needs >10% of the damage mask inside the vehicle.
                if vehicle_mask is not None:
                    overlap_area = float(np.logical_and(mask, vehicle_mask).sum())
                    if (overlap_area / damage_area) <= 0.1:
                        continue

                detections.append(
                    Detection(
                        class_name=damage_config.CLASS_NAMES[class_id],
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        polygon=self._mask_to_polygon(mask),
                        area_px=damage_area,
                        confidence=float(conf),
                    )
                )

        # Highest confidence first.
        detections.sort(key=lambda d: d.confidence, reverse=True)

        # Annotated overlay, written where the caller asked.
        annotated_path: Optional[Path] = None
        if annotate and annotated_dir is not None:
            out_dir = Path(annotated_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = out_dir / f"{Path(source).stem}_annotated.jpg"
            cv2.imwrite(str(annotated_path), damage_results.plot())  # BGR array

        return DetectionResult(
            detections=detections,
            image_path=Path(source),
            annotated_path=annotated_path,
            vehicle_area_px=vehicle_area_px,
            vehicle_mask=vehicle_mask,
            image_width=int(img_w),
            image_height=int(img_h),
            is_stub=False,
            model_name=self.model_name,
        )

    def _mask_to_polygon(self, mask: np.ndarray) -> List[Point]:
        """Convert a binary mask to a simplified polygon contour using OpenCV."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return []

        # Largest contour is the damage region.
        largest = max(contours, key=cv2.contourArea)

        # Approximate to reduce the number of points.
        epsilon = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        return [(float(p[0][0]), float(p[0][1])) for p in approx]


class StubDetector:
    """Returns fixed detections for testing without trained weights."""

    def __init__(self):
        self.weights = Path("weights/best.pt")
        self.model_name = "StubDetector"
        self.enable_vehicle_detection = False

    def detect(
        self,
        source: str | Path,
        *,
        augment: bool = False,
        annotate: bool = True,
        annotated_dir: Optional[Path | str] = None,
    ) -> DetectionResult:
        """Return the same two regions for every photograph."""
        w, h = 640, 480  # defaults, overridden by the real image size if readable
        try:
            from PIL import Image
            with Image.open(source) as img:
                w, h = img.size
        except Exception:
            pass

        # Two regions covering ~1% and ~0.96% of the frame, chosen to exercise
        # both severity bands.
        detections = [
            Detection(
                class_name="scratch",
                bbox=(w * 0.1, h * 0.1, w * 0.3, h * 0.3),
                polygon=[
                    (w * 0.1, h * 0.1),
                    (w * 0.3, h * 0.1),
                    (w * 0.3, h * 0.3),
                    (w * 0.1, h * 0.3),
                ],
                area_px=w * h * 0.01,
                confidence=0.9,
            ),
            Detection(
                class_name="dent",
                bbox=(w * 0.6, h * 0.2, w * 0.9, h * 0.5),
                polygon=[
                    (w * 0.6, h * 0.2),
                    (w * 0.9, h * 0.2),
                    (w * 0.9, h * 0.5),
                    (w * 0.6, h * 0.5),
                ],
                area_px=w * h * 0.0096,
                confidence=0.85,
            ),
        ]

        return DetectionResult(
            detections=detections,
            image_path=Path(source),
            annotated_path=None,
            vehicle_area_px=0.0,
            vehicle_mask=None,
            image_width=w,
            image_height=h,
            is_stub=True,
            model_name=self.model_name,
        )


def get_detector(
    weights: str | Path = "weights/best.pt",
    vehicle_weights: str | Path = "yolov8n-seg.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    enable_vehicle_detection: bool = True,
) -> YoloSegDetector | StubDetector:
    """Get a damage detector instance, falling back to the stub if weights are missing."""
    weights_path = Path(weights)
    if weights_path.is_file():
        return YoloSegDetector(
            weights=weights_path,
            vehicle_weights=vehicle_weights,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            enable_vehicle_detection=enable_vehicle_detection,
        )
    return StubDetector()