"""Severity assessment from segmentation mask area.

Why this is a rule and not a model
----------------------------------
The CarDD-derived dataset labels *what* the damage is, never *how bad* it is.
There are no severity labels, so there is nothing to train a severity
classifier on. Anyone claiming a learned severity model here is either using a
different dataset or overstating what they built.

What we do instead is defensible and inspectable: the trained segmentation
model outputs a polygon per damage, a polygon has an area, and area is a
reasonable proxy for how much work the repair needs. For linear damages like
scratches and cracks, length is a better proxy than area for repair effort.
The mapping from metric to Minor / Moderate / Severe is a documented rule in
damage_config.SEVERITY_RULES, tightened by domain knowledge (a shattered
windscreen is never "minor" - the glass gets replaced either way).

The honest limitation, and have this answer ready
------------------------------------------------
Area is measured relative to a reference area, and by default that reference is
the whole image. So the same dent photographed from two metres and from twenty
centimetres yields two different area fractions and possibly two different
verdicts. Scale-invariance would need the car itself as the reference.

That is why every function here takes `reference_area` rather than reaching for
the image size itself: a COCO-pretrained segmentation model already knows the
`car` class, so masking the vehicle and passing its area in makes the whole
pipeline scale-invariant without touching this file. See
`estimate_reference_area` for the hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .damage_config import SEVERITY_ORDER, SEVERITY_RULES

Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def polygon_area(points: Sequence[Point]) -> float:
    """Area of a simple polygon, by the shoelace formula.

    Sum the cross products of consecutive vertex pairs and halve the absolute
    value. Returns 0.0 for degenerate input (fewer than three vertices), which
    happens in a handful of dataset labels.

    Units follow the input: pass pixel coordinates for pixel area, or normalised
    0-1 coordinates for a fraction of the image. Prefer counting the binary mask
    pixels when the detector gives you them - it is exact even for
    self-intersecting polygons, where the shoelace formula quietly cancels the
    overlapping region against itself.
    """
    n = len(points)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def polygons_area(polygons: Iterable[Sequence[Point]]) -> float:
    """Combined area of several polygons.

    This is a plain sum, so overlapping polygons are double-counted. That is the
    right behaviour for our use: two overlapping scratch polygons are two
    annotations of damage, and the union would understate the work. Where it
    would be wrong - aggregating one class across a panel - the pipeline caps
    the fraction at 1.0 instead.
    """
    return sum(polygon_area(p) for p in polygons)


def estimate_reference_area(image_width: int, image_height: int,
                            vehicle_area_px: float | None = None) -> float:
    """The denominator for area fractions.

    Pass `vehicle_area_px` (the mask area of the car itself, from a
    COCO-pretrained segmentation model) to make severity independent of how
    close the camera was. Without it we fall back to the full frame, which is
    the documented limitation above.
    """
    frame = float(image_width * image_height)
    if vehicle_area_px and vehicle_area_px > 0:
        # Guard against a bad vehicle mask making the fraction explode: the car
        # cannot occupy less than ~5% of a photograph taken to show its damage.
        return max(float(vehicle_area_px), 0.05 * frame)
    return frame


# ---------------------------------------------------------------------------
# Banding
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeverityVerdict:
    severity: str          # one of damage_config.SEVERITY_ORDER
    area_fraction: float   # damage area / reference area, clamped to 0-1
    reason: str            # human-readable, goes straight into the PDF report


def _approximate_length(polygon: Sequence[Point]) -> float:
    """Approximate length of a polygon (heuristic for open contours like scratches/cracks).
    
    For closed contours, this returns perimeter.
    For open contours, this approximates the length by treating the polygon as
    a sequence of connected line segments.
    """
    if len(polygon) < 2:
        return 0.0
    
    length = 0.0
    for i in range(len(polygon) - 1):
        x1, y1 = polygon[i]
        x2, y2 = polygon[i + 1]
        length += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    # For closed polygons, add the closing segment
    if len(polygon) >= 3:
        x1, y1 = polygon[-1]
        x2, y2 = polygon[0]
        length += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        
    return length


def classify(class_name: str, area_fraction: float,
             polygon: list[tuple[float, float]] = None,
             bbox: tuple[float, float, float, float] = None,
             image_width: int = 0, image_height: int = 0) -> SeverityVerdict:
    """Band a damage instance into severity using class-appropriate metrics.
    
    Uses different geometric features per class where area alone is insufficient:
    - scratch/crack: approximate length (better correlates with repaint effort)
    - dent/dislocated_part: area (often reasonable for these)
    - others: area fraction (default)
    """
    rule = SEVERITY_RULES.get(class_name)
    if rule is None:
        raise KeyError(
            f"no severity rule for {class_name!r}; "
            f"known classes: {sorted(SEVERITY_RULES)}"
        )

    # Store the original area fraction for the return value
    original_area_fraction = area_fraction
    
    # --- SELECT APPROPRIATE METRIC PER CLASS ---
    metric = area_fraction  # default fallback
    metric_description = "area fraction"  # For reason text
    
    if polygon and len(polygon) >= 3:
        if class_name in ["scratch", "crack"]:
            # For linear damages: use approximate length (proxy for repaint length)
            length = _approximate_length(polygon)
            # Normalize by characteristic size: prefer image dimensions, then bbox
            if image_width > 0 and image_height > 0:
                characteristic_size = (image_width**2 + image_height**2)**0.5
                metric_description = f"length / image diagonal ({characteristic_size:.0f}px)"
            elif bbox is not None:
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                characteristic_size = (bbox_width**2 + bbox_height**2)**0.5
                characteristic_size = max(characteristic_size, 1.0)
                metric_description = f"length / bbox diagonal ({characteristic_size:.0f}px)"
            else:
                # Fallback: use area-based normalization (should not happen if we have polygon)
                characteristic_size = 100.0  # reasonable default
                metric_description = f"length / nominal size ({characteristic_size:.0f}px)"
            metric = min(length / characteristic_size, 1.0)
            
        elif class_name == "dent":
            # For dents: area often works reasonably well
            # Keeping area_fraction as base
            pass  # metric remains area_fraction
            
        elif class_name == "dislocated_part":
            # For missing parts: area is reasonable starting point
            pass  # metric remains area_fraction
            
        # glass_shatter, lamp_broken, tire_flat: area_fraction usually adequate

    # --- SEVERITY BANDING (unchanged logic, using new 'metric') ---
    fraction = min(max(float(metric), 0.0), 1.0)

    if fraction <= rule.minor_max:
        banded = "minor"
    elif fraction <= rule.moderate_max:
        banded = "moderate"
    else:
        banded = "severe"

    floored = SEVERITY_ORDER[max(SEVERITY_ORDER.index(banded),
                                 SEVERITY_ORDER.index(rule.floor))]

    # Generate reason text that accurately reflects what was measured
    if class_name in ["scratch", "crack"] and polygon and len(polygon) >= 3:
        # For linear damages, describe the length metric
        length = _approximate_length(polygon)
        if image_width > 0 and image_height > 0:
            characteristic_size = (image_width**2 + image_height**2)**0.5
        elif bbox is not None:
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            characteristic_size = (bbox_width**2 + bbox_height**2)**0.5
            characteristic_size = max(characteristic_size, 1.0)
        else:
            characteristic_size = 100.0
        
        normalized_length = length / characteristic_size
        pct = normalized_length * 100
        
        if floored != banded:
            reason = (
                f"length metric {pct:.2f}% of reference; raised to {floored} "
                f"because {rule.note.lower()}"
            )
        else:
            reason = f"length metric {pct:.2f}% of reference"
    else:
        # For other classes, use area fraction as before
        pct = fraction * 100
        if floored != banded:
            reason = (
                f"covers {pct:.2f}% of the reference area; raised to "
                f"{floored} because {rule.note.lower()}"
            )
        else:
            reason = f"covers {pct:.2f}% of the reference area"

    return SeverityVerdict(severity=floored, area_fraction=original_area_fraction, reason=reason)


def worst(severities: Iterable[str]) -> str:
    """The most serious severity in a group, for the overall vehicle verdict.

    Max, not mean: a car with four minor scratches and one severe dent is a
    severely damaged car. Averaging would hide the expensive part, which for a
    system that outputs a repair bill is the wrong failure direction.
    """
    ranked = [SEVERITY_ORDER.index(s) for s in severities if s in SEVERITY_ORDER]
    if not ranked:
        return "minor"
    return SEVERITY_ORDER[max(ranked)]
