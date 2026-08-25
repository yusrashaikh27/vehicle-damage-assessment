"""Damage detection: a thin, testable wrapper around the trained YOLO11-seg model.

Two design decisions worth knowing about.

**The app runs before the weights exist.** Training takes hours on a free Colab
T4, so `get_detector()` falls back to `StubDetector` when `weights/best.pt` is
missing. The stub returns fixed, plausible detections, which means the whole
chain - upload, severity, cost, PDF, database, admin - can be built and proven
today and the real model dropped in as a single file later. The web page says
loudly when it is showing stub output, because a demo that silently shows fake
results is a trap.

**Mask area is measured by rasterising, not by the shoelace formula.**
Ultralytics gives polygons in `masks.xy`, and a shoelace area over those is
wrong whenever a polygon self-intersects - the overlapping region cancels
against itself and the area comes out too small, sometimes near zero. Since the
area feeds severity and severity feeds the bill, that failure mode is not
acceptable. Instead each polygon is drawn onto a blank single-channel image at
the original resolution and the filled pixels are counted. Exact, and it depends
on nothing but Pillow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .damage_config import CLASS_NAMES

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"

# Confidence floor for a detection to be reported. 0.35 is deliberately lower
# than the 0.5 people usually reach for: missing damage understates a repair
# bill, and an over-reported scratch is cheap to dismiss by eye, so recall is
# worth more than precision here. Revisit once the real PR curve exists.
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.50
DEFAULT_IMGSZ = 640


class WeightsNotFound(RuntimeError):
    pass


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]          # x1, y1, x2, y2 in pixels
    polygon: list[tuple[float, float]] = field(default_factory=list)
    area_px: float = 0.0


@dataclass
class DetectionResult:
    detections: list[Detection]
    image_width: int
    image_height: int
    annotated_path: Path | None = None
    is_stub: bool = False
    model_name: str = ""

    @property
    def image_area(self) -> float:
        return float(self.image_width * self.image_height)


# ---------------------------------------------------------------------------
# Area measurement
# ---------------------------------------------------------------------------

def rasterised_area(polygon: Sequence[tuple[float, float]],
                    width: int, height: int) -> float:
    """Filled pixel count of a polygon drawn at (width, height).

    Correct for self-intersecting polygons, where the shoelace formula is not.
    Uses Pillow's even-odd polygon fill.
    """
    if len(polygon) < 3:
        return 0.0
    from PIL import Image, ImageDraw

    canvas = Image.new("1", (max(width, 1), max(height, 1)), 0)
    ImageDraw.Draw(canvas).polygon([(float(x), float(y)) for x, y in polygon],
                                   fill=1)
    # A "1"-mode histogram is [zeros, ones]; the ones are the filled pixels.
    return float(canvas.histogram()[-1])


# ---------------------------------------------------------------------------
# Real detector
# ---------------------------------------------------------------------------

class YoloSegDetector:
    """Wraps ultralytics YOLO11-seg. The model is loaded once, on first use."""

    def __init__(self, weights: Path | str = DEFAULT_WEIGHTS,
                 conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
                 imgsz: int = DEFAULT_IMGSZ):
        self.weights = Path(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self._model = None

    @property
    def available(self) -> bool:
        return self.weights.exists()

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.available:
            raise WeightsNotFound(
                f"No trained weights at {self.weights}. Train in Colab with "
                f"notebooks/train_colab.ipynb, then copy best.pt there."
            )
        # Imported here, not at module scope: ultralytics pulls in torch, which
        # takes seconds and hundreds of MB. Django's autoreloader imports every
        # module on every code change, so a top-level import would make the dev
        # server painful to work with.
        from ultralytics import YOLO

        logger.info("loading segmentation weights from %s", self.weights)
        self._model = YOLO(str(self.weights))
        return self._model

    def detect(self, image_path: Path | str,
               annotated_dir: Path | str | None = None) -> DetectionResult:
        model = self._load()
        image_path = Path(image_path)

        results = model.predict(
            source=str(image_path), imgsz=self.imgsz, conf=self.conf,
            iou=self.iou, verbose=False,
        )
        result = results[0]
        height, width = result.orig_shape

        # Ultralytics names come from the weights file, so a model trained on a
        # different taxonomy cannot silently be misread as ours.
        names = result.names if isinstance(result.names, dict) else dict(
            enumerate(result.names))

        detections: list[Detection] = []
        boxes = result.boxes
        polygons = result.masks.xy if result.masks is not None else []

        for i in range(len(boxes) if boxes is not None else 0):
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())

            polygon: list[tuple[float, float]] = []
            if i < len(polygons):
                polygon = [(float(px), float(py)) for px, py in polygons[i]]

            if polygon:
                area = rasterised_area(polygon, width, height)
            else:
                # A segmentation model that returned no mask for a box is
                # anomalous; fall back to the box area rather than dropping the
                # detection, and note it so it is not mistaken for a real mask.
                area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
                logger.warning("detection %d has no mask; used box area", i)

            detections.append(Detection(
                class_id=class_id,
                class_name=names.get(class_id, f"class_{class_id}"),
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                polygon=polygon,
                area_px=area,
            ))

        annotated_path = None
        if annotated_dir is not None:
            annotated_path = self._save_annotated(result, image_path,
                                                  Path(annotated_dir))

        return DetectionResult(
            detections=detections, image_width=width, image_height=height,
            annotated_path=annotated_path, is_stub=False,
            model_name=self.weights.name,
        )

    @staticmethod
    def _save_annotated(result, image_path: Path, annotated_dir: Path) -> Path:
        """Write the masked, labelled image. This is the figure users see."""
        from PIL import Image

        annotated_dir.mkdir(parents=True, exist_ok=True)
        out = annotated_dir / f"{image_path.stem}_annotated.jpg"
        # result.plot() returns a BGR array (an OpenCV convention); PIL expects
        # RGB, so the channel order is reversed. Skip this and every red mask
        # comes out blue.
        Image.fromarray(result.plot()[:, :, ::-1]).save(out, quality=90)
        return out


# ---------------------------------------------------------------------------
# Stub detector
# ---------------------------------------------------------------------------

class StubDetector:
    """Fixed, plausible detections so the app works before training finishes.

    Draws two rectangles roughly where damage tends to sit in a photograph: a
    scratch across about 3% of the frame (moderate) and a dent over about 0.7%
    (minor). Between them they exercise both severity bands, the painting and
    non-painting cost paths, and the aggregation logic.
    """

    available = True

    def detect(self, image_path: Path | str,
               annotated_dir: Path | str | None = None) -> DetectionResult:
        from PIL import Image, ImageDraw

        image_path = Path(image_path)
        with Image.open(image_path) as im:
            width, height = im.size
            annotated = im.convert("RGB").copy()

        spec = [
            ("scratch", 0.35, 0.55, 0.55, 0.60, 0.88),
            ("dent",    0.20, 0.30, 0.32, 0.38, 0.74),
        ]

        detections: list[Detection] = []
        draw = ImageDraw.Draw(annotated)
        for class_name, fx1, fy1, fx2, fy2, conf in spec:
            x1, y1 = fx1 * width, fy1 * height
            x2, y2 = fx2 * width, fy2 * height
            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            draw.polygon(polygon, outline=(255, 60, 60), width=3)
            draw.text((x1 + 4, max(y1 - 14, 0)), f"{class_name} (stub)",
                      fill=(255, 60, 60))
            detections.append(Detection(
                class_id=CLASS_NAMES.index(class_name),
                class_name=class_name,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                polygon=polygon,
                area_px=rasterised_area(polygon, width, height),
            ))

        annotated_path = None
        if annotated_dir is not None:
            annotated_dir = Path(annotated_dir)
            annotated_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = annotated_dir / f"{image_path.stem}_annotated.jpg"
            annotated.save(annotated_path, quality=90)

        return DetectionResult(
            detections=detections, image_width=width, image_height=height,
            annotated_path=annotated_path, is_stub=True,
            model_name="stub (no trained weights found)",
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_cached: YoloSegDetector | StubDetector | None = None


def get_detector(weights: Path | str = DEFAULT_WEIGHTS,
                 force_reload: bool = False):
    """The real detector if weights are present, otherwise the stub.

    Cached at module level so the ~50 MB model is loaded once per process
    rather than once per request.
    """
    global _cached
    if _cached is not None and not force_reload:
        return _cached
    real = YoloSegDetector(weights)
    if real.available:
        _cached = real
    else:
        logger.warning("no weights at %s - falling back to StubDetector", weights)
        _cached = StubDetector()
    return _cached
