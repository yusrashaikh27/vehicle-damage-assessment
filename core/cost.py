"""Repair cost estimation - repair only, and a range at every level.

The formula, in words
--------------------
For each distinct damage type found on the vehicle, on its component:

    line range = panel repair range
                 x  where the severity sits inside it     (REPAIR_ACTIONS shares)
                 x  segment price multiplier
                 x  instance factor (repeated damage of the same type)

Every line is a (low, high) pair, the subtotal is the sum of the lows and the sum
of the highs, GST is added to each end, and the presented total is that range
rounded outward to a round figure. Nothing is ever shown as a single number.

The price list is itself a low-high range per panel, so no extra percentage is
added around the total: the source range already is the uncertainty. (The old
model built one calculated figure and widened only the total afterwards.)

Decisions that are easy to get wrong
------------------------------------

1. **A panel is prepared and painted once.** If one door has a dent and a crack,
   the dearest damage is charged in full and the others at INSTANCE_FACTOR (40%),
   because masking, prep and paint-booth time are shared. Lamps and glass are the
   exception: they are replaced, so each is a separate part.

2. **Repeated damage of one type adds up, but sublinearly.** Two dents on one door
   take longer than one, yet prep and paint are shared. The second and later
   instances are charged at 40% (INSTANCE_FACTOR), capped at INSTANCE_FACTOR_CAP,
   so a noisy detection with eleven overlapping scratch polygons cannot produce an
   absurd bill. 40% is a judgement, not a measurement.

3. **A multi-panel entry is priced for every panel it covers.** The "front + rear
   door" range in damage_config is already the two doors added together.

4. **Severity places a line inside the panel's range.** Minor damage sits in the
   lower part, severe in the upper part (see REPAIR_ACTIONS). The slices are a
   judgement, so the report says so.

Repair only
-----------
The project estimates repair, not replacement. Lamps and glass cannot be repaired,
so they are priced from the list's replacement table and labelled as replacement;
a flat tyre is not in the list and is left unpriced with a note.

Why the user-selected panel is sometimes overridden
---------------------------------------------------
The user picks the panel at upload because no dataset class names a component.
But three classes *do* imply their own component - a broken lamp is a lamp
whatever panel you photographed, glass is glass, a flat tyre is a tyre - so for
those we price the implied component instead. See COMPONENT_OVERRIDE.

Pricing across several photographs
----------------------------------
An inspection carries a walk-around: front, front-left, left, and so on. Each
photograph is detected separately, but they must be priced *together*, because
the decisions above are properties of the vehicle, not of a photograph.
`estimate_combined` therefore merges the views first and prices once.

Merging takes the **maximum** of the per-view figures rather than the sum. That is
the conservative reading - it can understate genuinely separate damage on one
component, but it cannot invent damage that is not there, and an estimate that
quietly doubles because the owner took two photographs of one scratch would be
indefensible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .damage_config import (
    CLASS_LABELS,
    COST_SOURCE,
    CURRENCY,
    CURRENCY_SYMBOL,
    DEFAULT_PANEL,
    DEFAULT_SEGMENT,
    GST_RATE,
    PANELS,
    RANGE_LINE_STEP,
    RANGE_STEP_TIERS,
    REPAIR_ACTIONS,
    SEGMENTS,
    SEVERITY_LABELS,
    Panel,
    RepairAction,
)
from .severity import worst

# Classes that name their own component, overriding whatever panel the user
# picked. The value is the fallback; if the user already selected a component of
# the right kind we keep theirs, because they can see the photograph and we
# cannot tell a headlamp from a taillamp from the damage class alone.
COMPONENT_OVERRIDE: dict[str, tuple[str, tuple[str, ...]]] = {
    #  class            fallback component   components we accept from the user
    "lamp_broken":   ("headlamp",   ("headlamp", "taillamp")),
    "glass_shatter": ("windscreen", ("windscreen", "rear_glass", "door_glass")),
    "tire_flat":     ("wheel",      ("wheel",)),
}

# Second and subsequent instances of the same damage type are charged at this
# fraction, up to INSTANCE_FACTOR_CAP in total. The same fraction is used for a
# second damage type on the same panel (shared preparation and paint).
INSTANCE_FACTOR = 0.40
INSTANCE_FACTOR_CAP = 2.5


@dataclass(frozen=True)
class DamageGroup:
    """One damage type on the vehicle, after the pipeline has aggregated it."""
    class_name: str
    severity: str
    instances: int = 1
    area_fraction: float = 0.0
    confidence: float = 0.0


@dataclass(frozen=True)
class CostLine:
    """One priced line. `low` and `high` are rupees, before GST."""
    class_name: str
    class_label: str
    severity: str
    severity_label: str
    component: str
    component_label: str
    action: str
    instances: int
    low: float
    high: float
    replacement: bool = False

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2


@dataclass(frozen=True)
class Quotation:
    lines: list[CostLine] = field(default_factory=list)
    panel: str = DEFAULT_PANEL
    segment: str = DEFAULT_SEGMENT
    # Every panel that contributed damage. For a single-photograph estimate this
    # is just (panel,); for a walk-around it lists each angle's panel.
    panels: tuple[str, ...] = ()
    # Sums of the (already rounded) lines, and GST on each end. Not rounded
    # further, so the page can show that lines add up to the subtotal.
    subtotal_low: float = 0.0
    subtotal_high: float = 0.0
    gst_low: float = 0.0
    gst_high: float = 0.0
    # The presented estimate: GST-inclusive, rounded outward to a round figure.
    # Both are 0.0 when nothing was priced.
    low: float = 0.0
    high: float = 0.0
    # Empty means "nothing was graded", which is the truthful default for a
    # quotation nobody has priced yet. Both the results page and the PDF test
    # this for falsiness to decide between a severity badge and "None".
    overall_severity: str = ""
    currency: str = CURRENCY
    notes: list[str] = field(default_factory=list)

    # Midpoints, for the few places that still want one figure to store (the
    # deprecated single-figure columns). Never presented as the estimate.
    @property
    def subtotal(self) -> float:
        return (self.subtotal_low + self.subtotal_high) / 2

    @property
    def gst(self) -> float:
        return (self.gst_low + self.gst_high) / 2

    @property
    def total(self) -> float:
        return ((self.subtotal_low + self.gst_low)
                + (self.subtotal_high + self.gst_high)) / 2


def resolve_component(class_name: str, user_panel: str) -> str:
    """Which component's price list applies to this damage."""
    override = COMPONENT_OVERRIDE.get(class_name)
    if override is None:
        return user_panel if user_panel in PANELS else DEFAULT_PANEL
    fallback, accepted = override
    return user_panel if user_panel in accepted else fallback


def instance_factor(instances: int) -> float:
    """Multiplier for repeated damage of the same type."""
    if instances <= 1:
        return 1.0
    return min(1.0 + INSTANCE_FACTOR * (instances - 1), INSTANCE_FACTOR_CAP)


def range_step_for(total: float) -> int:
    """Rounding step for a total: the step of the last tier it reaches.

    Rs. 500 below Rs. 20,000 and Rs. 1,000 from there up (RANGE_STEP_TIERS), so
    the ends of the range are round figures at the scale of the estimate.
    """
    step = RANGE_STEP_TIERS[0][1]
    for threshold, tier_step in RANGE_STEP_TIERS:
        if total >= threshold:
            step = tier_step
    return step


def range_step_text() -> str:
    """The rounding rule in words, for the notes and the report."""
    parts = []
    for i, (threshold, step) in enumerate(RANGE_STEP_TIERS):
        if i + 1 < len(RANGE_STEP_TIERS):
            parts.append(f"Rs. {step:,} below Rs. {int(RANGE_STEP_TIERS[i + 1][0]):,}")
        else:
            parts.append(f"Rs. {step:,} from Rs. {int(threshold):,} up")
    return ", ".join(parts)


def _round_down(amount: float, step: float) -> float:
    return float(math.floor(amount / step) * step)


def _round_up(amount: float, step: float) -> float:
    return float(math.ceil(amount / step) * step)


def _action_range(action: RepairAction, comp: Panel,
                  instances: int) -> tuple[float, float] | None:
    """(low, high) for one action on one component, before the segment
    multiplier. None when the price list has no figure for it."""
    if action.mode == "unpriced":
        return None

    if action.mode == "replace":
        if comp.replace_high <= 0:
            return None
        # One part is bought however many instances the model found.
        return comp.replace_low, comp.replace_high

    # Both remaining modes are repairs, so the component must have a repair price.
    if comp.repair_high <= 0:
        return None
    factor = instance_factor(instances)

    if action.mode == "flat":
        n = comp.panel_count
        return action.flat_low * n * factor, action.flat_high * n * factor

    span = comp.repair_high - comp.repair_low
    return ((comp.repair_low + span * action.low_share) * factor,
            (comp.repair_low + span * action.high_share) * factor)


def estimate(groups: list[DamageGroup],
             panel: str = DEFAULT_PANEL,
             segment: str = DEFAULT_SEGMENT) -> Quotation:
    """Price a set of aggregated damages found on one panel.

    `groups` must already be aggregated - one entry per damage class, with the
    severity decided. core.pipeline does that; this function only does money, so
    that the cost model can be unit-tested without a model, an image, or a
    database.

    A thin wrapper: it resolves each group's component from the one panel given,
    then hands the pairs to `_price`.
    """
    pairs = [(resolve_component(g.class_name, panel), g) for g in groups]
    return _price(pairs, segment=segment, panel=panel, panels=(panel,))


def estimate_combined(views: list[tuple[str, list[DamageGroup]]],
                      segment: str = DEFAULT_SEGMENT) -> Quotation:
    """Price damage seen across several photographs of one vehicle.

    `views` is one (panel, groups) pair per photograph: the panel the owner said
    that shot contains, and the groups the pipeline found in it.

    Merging happens before pricing, keyed on (damage class, component), because
    "a panel is prepared once" is a fact about the vehicle and would be broken by
    pricing each photograph on its own.

    Where a class/component pair appears in more than one view, the merged group
    takes the worst severity but the **maximum** instance count and area, not the
    sum. See the module docstring.
    """
    merged: dict[tuple[str, str], DamageGroup] = {}
    seen_in: dict[tuple[str, str], int] = {}

    for view_panel, groups in views:
        for group in groups:
            component = resolve_component(group.class_name, view_panel)
            key = (group.class_name, component)
            seen_in[key] = seen_in.get(key, 0) + 1
            previous = merged.get(key)
            if previous is None:
                merged[key] = group
            else:
                merged[key] = DamageGroup(
                    class_name=group.class_name,
                    severity=worst([previous.severity, group.severity]),
                    instances=max(previous.instances, group.instances),
                    area_fraction=max(previous.area_fraction, group.area_fraction),
                    confidence=max(previous.confidence, group.confidence),
                )

    # Deterministic order regardless of which angle was uploaded first, matching
    # the convention in core.pipeline.aggregate.
    pairs = sorted(
        ((component, group) for (_, component), group in merged.items()),
        key=lambda pair: (-pair[1].area_fraction, pair[1].class_name, pair[0]),
    )

    # Say out loud wherever a merge changed the arithmetic, naming the component,
    # so a reader can see which figure was held down and why.
    notes: list[str] = []
    for component in sorted({c for (_, c), n in seen_in.items() if n > 1}):
        label = PANELS[component].label if component in PANELS else component
        notes.append(
            f"{label}: the same damage type appears in more than one photograph. "
            f"The largest single measurement is used rather than the sum, because "
            f"overlapping angles may show one instance twice."
        )

    panels = tuple(panel for panel, _ in views)
    # `panel` only means something when every view shares one. Otherwise the
    # honest answer is that there is no single panel, and `panels` carries the
    # detail.
    distinct = set(panels)
    single = panels[0] if len(distinct) == 1 else DEFAULT_PANEL

    return _price(pairs, segment=segment, panel=single, panels=panels,
                  extra_notes=notes)


def _price(pairs: list[tuple[str, DamageGroup]],
           segment: str,
           panel: str,
           panels: tuple[str, ...],
           extra_notes: tuple[str, ...] | list[str] = ()) -> Quotation:
    """The money. One (component, group) pair per damage type to be priced.

    Taking pre-resolved components rather than a panel is what lets one estimate
    span several photographs while still charging each component once.
    """
    seg = SEGMENTS.get(segment) or SEGMENTS[DEFAULT_SEGMENT]
    mult = seg.price_multiplier
    notes: list[str] = []

    if not pairs:
        return Quotation(
            panel=panel, segment=segment, panels=panels,
            # Empty, not "minor". Nothing was detected, so there is nothing to
            # grade, and "minor" would be a claim the measurement does not
            # support.
            overall_severity="",
            notes=["No damage detected, so no repair cost is estimated."]
                  + list(extra_notes),
        )

    overall = worst(g.severity for _, g in pairs)

    # Pass 1: a (low, high) per damage group, grouped by component.
    priced: dict[str, list[tuple[float, float, DamageGroup, RepairAction]]] = {}
    for component, group in pairs:
        action = REPAIR_ACTIONS.get((group.class_name, group.severity))
        label = CLASS_LABELS.get(group.class_name, group.class_name)
        if action is None:
            notes.append(
                f"No repair action defined for {group.class_name} / "
                f"{group.severity}; skipped."
            )
            continue
        comp = PANELS.get(component) or PANELS[DEFAULT_PANEL]
        rng = _action_range(action, comp, group.instances)
        if rng is None:
            notes.append(
                f"{label} on {comp.label.lower()}: not covered by the price "
                f"list, so it is not included in this estimate."
            )
            continue
        priced.setdefault(component, []).append(
            (rng[0] * mult, rng[1] * mult, group, action))

    # Pass 2: build the lines. On one component the dearest damage is charged in
    # full and the rest at INSTANCE_FACTOR (the panel is prepared and painted
    # once). A replaced part is always charged in full.
    lines: list[CostLine] = []
    shared_used = False
    for component, items in priced.items():
        comp = PANELS.get(component) or PANELS[DEFAULT_PANEL]
        items.sort(key=lambda t: t[1], reverse=True)       # dearest first
        for rank, (lo, hi, group, action) in enumerate(items):
            replacement = action.mode == "replace"
            share = 1.0 if (rank == 0 or replacement) else INSTANCE_FACTOR
            if share < 1.0:
                shared_used = True
            lines.append(CostLine(
                class_name=group.class_name,
                class_label=CLASS_LABELS.get(group.class_name, group.class_name),
                severity=group.severity,
                severity_label=SEVERITY_LABELS.get(group.severity, group.severity),
                component=component,
                component_label=comp.label,
                action=action.description,
                instances=group.instances,
                low=_round_down(lo * share, RANGE_LINE_STEP),
                high=_round_up(hi * share, RANGE_LINE_STEP),
                replacement=replacement,
            ))

    lines.sort(key=lambda l: (-l.high, l.class_name, l.component))

    if not lines:
        return Quotation(
            panel=panel, segment=segment, panels=panels,
            overall_severity=overall,
            notes=notes + ["Nothing in this inspection could be priced from the "
                           "price list, so no cost range is given."]
                  + list(extra_notes),
        )

    subtotal_low = sum(l.low for l in lines)
    subtotal_high = sum(l.high for l in lines)
    gst_low = round(subtotal_low * GST_RATE, 2)
    gst_high = round(subtotal_high * GST_RATE, 2)
    total_low = subtotal_low + gst_low
    total_high = subtotal_high + gst_high
    step = range_step_for(total_high)
    low = _round_down(total_low, step)
    high = _round_up(total_high, step)

    # ASCII "Rs." on purpose: these notes are stored and printed in the PDF, whose
    # built-in font has no rupee-sign glyph (it would print as a black box).
    notes.append(
        f"Every price is a range taken from the rate card: {COST_SOURCE} "
        f"Where a line falls inside its range follows the severity (minor in the "
        f"lower part, severe in the upper part); that placement is a judgement, "
        f"not a measured figure."
    )
    if any(l.instances > 1 for l in lines):
        notes.append(
            f"Repeated damage of the same type is charged at {INSTANCE_FACTOR:.0%} "
            f"for each instance after the first, because preparation and paint "
            f"time are shared."
        )
    if shared_used:
        notes.append(
            f"Where one panel has several damage types, the dearest is charged in "
            f"full and the others at {INSTANCE_FACTOR:.0%}, because the panel is "
            f"prepared and painted once."
        )
    multi_panel = sorted(
        {l.component for l in lines
         if PANELS.get(l.component) and PANELS[l.component].panel_count > 1}
    )
    for component in multi_panel:
        notes.append(
            f"{PANELS[component].label}: priced as "
            f"{PANELS[component].panel_count} panels, so the range covers both."
        )
    if any(l.replacement for l in lines):
        notes.append(
            "Lamp and glass damage cannot be repaired, so those lines are "
            "replacement prices. Everything else is repair only."
        )
    notes.append(
        f"The total includes GST at {GST_RATE:.0%} on both ends of the range and "
        f"is rounded outward to round figures ({range_step_text()})."
    )

    return Quotation(
        lines=lines,
        panel=panel,
        segment=segment,
        panels=panels,
        subtotal_low=subtotal_low,
        subtotal_high=subtotal_high,
        gst_low=gst_low,
        gst_high=gst_high,
        low=low,
        high=high,
        overall_severity=overall,
        notes=notes + list(extra_notes),
    )


def money(amount: float) -> str:
    """Format for display: Indian digit grouping, no decimals.

    Indian grouping puts the first separator after three digits from the right
    and every two thereafter, so 1234567 is 12,34,567 - not 1,234,567. Python's
    `:,` format cannot do this, hence the manual split.
    """
    whole = f"{int(round(amount)):d}"
    negative = whole.startswith("-")
    if negative:
        whole = whole[1:]
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return f"{'-' if negative else ''}{CURRENCY_SYMBOL}{whole}"


def money_range(low: float, high: float) -> str:
    """Format a low/high pair, e.g. '₹9,000 – ₹14,000'."""
    return f"{money(low)} – {money(high)}"