"""The assessment pipeline: image in, priced report out.

    detect -> measure area -> band severity -> aggregate -> price

This module is the only place the five stages meet, and it holds no Django
imports, so the whole system can be exercised from a plain Python prompt:

    from core.pipeline import assess
    print(assess("some_car.jpg", panel="front_bumper", segment="sedan").quotation.total)

Two pieces of real logic live here - aggregation within one photograph, and the
walk-around across several - and both are the difference between a plausible bill
and a silly one.

Aggregation
-----------
The model emits one detection per damage instance: five scratch polygons on a
door are five detections. Pricing each independently would charge for five
re-paints of one door. So detections are grouped by class, and each group's
severity is decided twice over, taking whichever is worse:

  * from the *summed* area of the group - because more total damage of one kind
    really is worse than the worst single patch, and
  * from the worst *individual* detection - so one severe gash is not diluted by
    being averaged against four light scuffs.

Taking the maximum of the two means neither reading can argue the other down.

The walk-around
---------------
A real inspection is several photographs of one car: front, front-left, left, and
so on. `assess_walkaround` runs detection on each one and then prices **once**,
across all of them, via `cost.estimate_combined`.

Pricing per photograph and adding the results up would be the obvious
implementation and it would be wrong: it buys two bumpers when the owner
photographs one bumper from two angles. Each view's own `Assessment` still
carries a `quotation`, because `assess` is also used on its own, but those
per-view quotations must never be summed - the walk-around's total is the one on
`WalkaroundAssessment.quotation`.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import severity as sev
from .cost import DamageGroup, Quotation, estimate, estimate_combined
from .damage_config import (
    CLASS_LABELS,
    DEFAULT_PANEL,
    DEFAULT_SEGMENT,
    SEVERITY_LABELS,
)
from .detector import Detection, DetectionResult, get_detector


@dataclass
class AssessedDetection:
    """One detection, with its severity decided."""
    class_name: str
    class_label: str
    confidence: float
    severity: str
    severity_label: str
    area_px: float
    area_fraction: float
    reason: str
    bbox: tuple[float, float, float, float]
    polygon: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Assessment:
    image_path: Path
    panel: str
    segment: str
    detections: list[AssessedDetection] = field(default_factory=list)
    groups: list[DamageGroup] = field(default_factory=list)
    quotation: Quotation = field(default_factory=Quotation)
    annotated_path: Path | None = None
    image_width: int = 0
    image_height: int = 0
    reference_area: float = 0.0
    is_stub: bool = False
    model_name: str = ""
    # Which walk-around position this photograph was taken from, when it came
    # from one. Empty for a standalone single-image assessment.
    angle: str = ""
    # How long this one photograph took, so the results page can report per-view
    # timing instead of only a total. Measured here rather than by the caller
    # because only this function knows where the work starts and stops.
    elapsed_ms: int = 0

    @property
    def damage_found(self) -> bool:
        return bool(self.detections)


def band_detections(detections: list[Detection],
                    reference_area: float) -> list[AssessedDetection]:
    """Give every detection an area fraction and a severity."""
    assessed: list[AssessedDetection] = []
    for det in detections:
        fraction = det.area_px / reference_area if reference_area > 0 else 0.0
        verdict = sev.classify(det.class_name, fraction)
        assessed.append(AssessedDetection(
            class_name=det.class_name,
            class_label=CLASS_LABELS.get(det.class_name, det.class_name),
            confidence=det.confidence,
            severity=verdict.severity,
            severity_label=SEVERITY_LABELS[verdict.severity],
            area_px=det.area_px,
            area_fraction=verdict.area_fraction,
            reason=verdict.reason,
            bbox=det.bbox,
            polygon=det.polygon,
        ))
    return assessed


def aggregate(assessed: list[AssessedDetection],
              reference_area: float) -> list[DamageGroup]:
    """Collapse detections into one group per damage class. See module docstring."""
    by_class: dict[str, list[AssessedDetection]] = defaultdict(list)
    for det in assessed:
        by_class[det.class_name].append(det)

    groups: list[DamageGroup] = []
    for class_name, dets in by_class.items():
        # Summed, then clamped in classify(): overlapping polygons of the same
        # class are double-counted by the sum, which is why the clamp to 1.0
        # matters rather than being defensive noise.
        total_fraction = sum(d.area_fraction for d in dets)
        from_total = sev.classify(class_name, total_fraction).severity
        from_worst = sev.worst(d.severity for d in dets)

        groups.append(DamageGroup(
            class_name=class_name,
            severity=sev.worst([from_total, from_worst]),
            instances=len(dets),
            area_fraction=min(total_fraction, 1.0),
            confidence=max(d.confidence for d in dets),
        ))

    # Deterministic order, so two runs on one image produce identical reports.
    groups.sort(key=lambda g: (-g.area_fraction, g.class_name))
    return groups


def assess(image_path: Path | str,
           panel: str = DEFAULT_PANEL,
           segment: str = DEFAULT_SEGMENT,
           annotated_dir: Path | str | None = None,
           detector=None,
           vehicle_area_px: float | None = None,
           angle: str = "") -> Assessment:
    """Run the full pipeline on one image."""
    started = time.perf_counter()
    image_path = Path(image_path)
    detector = detector or get_detector()

    result: DetectionResult = detector.detect(image_path,
                                              annotated_dir=annotated_dir)

    reference_area = sev.estimate_reference_area(
        result.image_width, result.image_height, vehicle_area_px)

    assessed = band_detections(result.detections, reference_area)
    groups = aggregate(assessed, reference_area)
    quotation = estimate(groups, panel=panel, segment=segment)

    # Worst first: the reader should meet the expensive damage before the scuffs.
    assessed.sort(key=lambda d: (-d.area_fraction, d.class_name))

    return Assessment(
        image_path=image_path,
        panel=panel,
        segment=segment,
        detections=assessed,
        groups=groups,
        quotation=quotation,
        annotated_path=result.annotated_path,
        image_width=result.image_width,
        image_height=result.image_height,
        reference_area=reference_area,
        is_stub=result.is_stub,
        model_name=result.model_name,
        angle=angle,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )


@dataclass
class WalkaroundAssessment:
    """Several photographs of one vehicle, and the single quotation for it."""
    segment: str
    views: list[Assessment] = field(default_factory=list)
    quotation: Quotation = field(default_factory=Quotation)
    is_stub: bool = False
    model_name: str = ""
    elapsed_ms: int = 0

    @property
    def detections(self) -> list[AssessedDetection]:
        """Every detection from every photograph, worst first.

        A flat tally of what the model output. Note that the same physical dent
        seen from two angles appears twice here - the *quotation* de-duplicates
        it, this list does not, because it describes detections and not damage.
        """
        flat = [det for view in self.views for det in view.detections]
        flat.sort(key=lambda d: (-d.area_fraction, d.class_name))
        return flat

    @property
    def damage_found(self) -> bool:
        return any(view.detections for view in self.views)

    @property
    def view_count(self) -> int:
        return len(self.views)


def assess_walkaround(shots: list[tuple[Path | str, str, str]],
                      segment: str = DEFAULT_SEGMENT,
                      annotated_dir: Path | str | None = None,
                      detector=None) -> WalkaroundAssessment:
    """Assess several photographs of one vehicle and price them together.

    `shots` is one (image_path, panel, angle) triple per photograph. Order is
    preserved in `views`, so a caller holding database rows can zip the two back
    together without needing an identifier here - which keeps this function free
    of anything Django-shaped.

    The pricing call is deliberately outside the loop. See the module docstring:
    a panel is bought once per *vehicle*, not once per photograph, so the views
    are merged first and priced as one job. Every photograph reuses the same
    detector instance, so the weights are loaded once however many angles arrive.
    """
    started = time.perf_counter()
    detector = detector or get_detector()

    views: list[Assessment] = []
    for index, (image_path, panel, angle) in enumerate(shots):
        # Each view gets its own subdirectory for its overlay. Without this, two
        # photographs that happen to share a filename would write the same
        # annotated path and the second would silently overwrite the first -
        # the results page would then show one angle's overlay under another
        # angle's heading, which is the kind of wrong that looks right.
        view_dir = None
        if annotated_dir is not None:
            view_dir = Path(annotated_dir) / f"view-{index}"
            view_dir.mkdir(parents=True, exist_ok=True)

        views.append(assess(
            image_path,
            panel=panel,
            segment=segment,
            annotated_dir=view_dir,
            detector=detector,
            angle=angle,
        ))

    quotation = estimate_combined(
        [(view.panel, view.groups) for view in views], segment=segment
    )

    # Provenance is a property of the detector, so it is the same for every view.
    # Taking it from the first is safe; `any`/`next` guards the no-photograph
    # case, which the form prevents but a direct caller could still reach.
    return WalkaroundAssessment(
        segment=segment,
        views=views,
        quotation=quotation,
        is_stub=any(view.is_stub for view in views),
        model_name=next((view.model_name for view in views), ""),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
