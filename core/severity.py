"""Severity assessment from segmentation mask area.

Why this is a rule and not a model
----------------------------------
The CarDD-derived dataset labels *what* the damage is, never *how bad* it is.
There are no severity labels, so there is nothing to train a severity
classifier on. Anyone claiming a learned severity model here is either using a
different dataset or overstating what they built.

What we do instead is defensible and inspectable: the trained segmentation
model outputs a polygon per damage, a polygon has an area, and area is a
reasonable proxy for how much work the repair needs. The mapping from area to
Minor / Moderate / Severe is a documented rule in damage_config.SEVERITY_RULES,
tightened by domain knowledge (a shattered windscreen is never "minor" - the
glass gets replaced either way).

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


def classify(class_name: str, area_fraction: float) -> SeverityVerdict:
    """Band an area fraction into a severity for one damage class.

    Two steps, in this order:

    1. Compare the area fraction against the class's own thresholds.
    2. Raise the result to the class's floor if it came out below it.

    Step 2 is what encodes "glass is always replaced". It can only ever raise
    the verdict, never lower it, so a large area is never argued away.
    """
    rule = SEVERITY_RULES.get(class_name)
    if rule is None:
        raise KeyError(
            f"no severity rule for {class_name!r}; "
            f"known classes: {sorted(SEVERITY_RULES)}"
        )

    fraction = min(max(float(area_fraction), 0.0), 1.0)

    if fraction <= rule.minor_max:
        banded = "minor"
    elif fraction <= rule.moderate_max:
        banded = "moderate"
    else:
        banded = "severe"

    floored = SEVERITY_ORDER[max(SEVERITY_ORDER.index(banded),
                                 SEVERITY_ORDER.index(rule.floor))]

    pct = fraction * 100
    if floored != banded:
        reason = (
            f"covers {pct:.2f}% of the reference area; raised to "
            f"{floored} because {rule.note.lower()}"
        )
    else:
        reason = f"covers {pct:.2f}% of the reference area"

    return SeverityVerdict(severity=floored, area_fraction=fraction, reason=reason)


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
