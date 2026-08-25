"""Repair cost estimation.

The formula, in words
--------------------
For each distinct damage type found on the vehicle:

    part      = component part cost x segment multiplier x replace_fraction
    labour    = segment labour rate x labour hours x instance factor
    paint     = component paint material x segment multiplier   (if it paints)

then GST on the sum. Every input comes from a table in damage_config.py.

Three modelling decisions that are easy to get wrong, and are the interesting
part of this module:

1. **A panel is bought once.** If the model finds both a severe crack and a
   severe dislocated part on the same bumper, both actions say "replace the
   part" - but you buy one bumper. Part cost therefore takes the *maximum*
   replace_fraction per component, never the sum.

2. **A panel is painted once.** Same argument: paint material is charged once
   per component, not once per scratch.

3. **Labour does add up, but sublinearly.** Two dents on one door genuinely take
   longer than one, yet the masking, prep and paint booth time is shared. The
   second and later instances are charged at 40% (INSTANCE_FACTOR), capped, so a
   noisy detection with eleven overlapping scratch polygons cannot produce an
   absurd bill.

Point 3 is the one to be candid about: 40% is a judgement, not a measurement.

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
the three decisions above are properties of the vehicle, not of a photograph. A
front shot and a front-left shot both showing the same bumper would otherwise buy
two bumpers - the very mistake point 1 exists to prevent, reintroduced one level
up. `estimate_combined` therefore merges the views first and prices once.

Merging raises a problem worth being candid about: the same physical dent
appearing in two overlapping photographs cannot be told apart from two separate
dents without re-identifying damage across views, which this system does not
attempt. Merging therefore takes the **maximum** of the per-view figures rather
than the sum. That is the conservative reading - it can understate genuinely
separate damage on one component, but it cannot invent damage that is not there,
and an estimate that quietly doubles because the owner took two photographs of
one scratch would be indefensible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .damage_config import (
    CLASS_LABELS,
    CURRENCY,
    CURRENCY_SYMBOL,
    DEFAULT_PANEL,
    DEFAULT_SEGMENT,
    GST_RATE,
    PANELS,
    REPAIR_ACTIONS,
    SEGMENTS,
    SEVERITY_LABELS,
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
# fraction of the labour hours, up to INSTANCE_FACTOR_CAP in total.
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
    class_name: str
    class_label: str
    severity: str
    severity_label: str
    component: str
    component_label: str
    action: str
    instances: int
    part_cost: float
    labour_cost: float
    labour_hours: float
    paint_cost: float

    @property
    def subtotal(self) -> float:
        return self.part_cost + self.labour_cost + self.paint_cost


@dataclass(frozen=True)
class Quotation:
    lines: list[CostLine] = field(default_factory=list)
    panel: str = DEFAULT_PANEL
    segment: str = DEFAULT_SEGMENT
    # Every panel that contributed damage. For a single-photograph estimate this
    # is just (panel,); for a walk-around it lists each angle's panel. `panel`
    # is kept as-is so single-panel callers are unaffected.
    panels: tuple[str, ...] = ()
    parts_total: float = 0.0
    labour_total: float = 0.0
    paint_total: float = 0.0
    subtotal: float = 0.0
    gst: float = 0.0
    total: float = 0.0
    # Empty means "nothing was graded", which is the truthful default for a
    # quotation nobody has priced yet. Django's SEVERITY_CHOICES field is
    # `blank=True` for the same reason, and both the results page and the PDF
    # test this for falsiness to decide between a severity badge and "None".
    overall_severity: str = ""
    currency: str = CURRENCY
    notes: list[str] = field(default_factory=list)


def resolve_component(class_name: str, user_panel: str) -> str:
    """Which component's price list applies to this damage."""
    override = COMPONENT_OVERRIDE.get(class_name)
    if override is None:
        return user_panel if user_panel in PANELS else DEFAULT_PANEL
    fallback, accepted = override
    return user_panel if user_panel in accepted else fallback


def instance_factor(instances: int) -> float:
    """Labour multiplier for repeated damage of the same type."""
    if instances <= 1:
        return 1.0
    return min(1.0 + INSTANCE_FACTOR * (instances - 1), INSTANCE_FACTOR_CAP)


def estimate(groups: list[DamageGroup],
             panel: str = DEFAULT_PANEL,
             segment: str = DEFAULT_SEGMENT) -> Quotation:
    """Price a set of aggregated damages found on one panel.

    `groups` must already be aggregated - one entry per damage class, with the
    severity decided. core.pipeline does that; this function only does money, so
    that the cost model can be unit-tested without a model, an image, or a
    database.

    A thin wrapper: it resolves each group's component from the one panel given,
    then hands the pairs to `_price`. Keeping it means every existing caller and
    test is unaffected by multi-view pricing.
    """
    pairs = [(resolve_component(g.class_name, panel), g) for g in groups]
    return _price(pairs, segment=segment, panel=panel, panels=(panel,))


def estimate_combined(views: list[tuple[str, list[DamageGroup]]],
                      segment: str = DEFAULT_SEGMENT) -> Quotation:
    """Price damage seen across several photographs of one vehicle.

    `views` is one (panel, groups) pair per photograph: the panel the owner said
    that shot contains, and the groups the pipeline found in it.

    Merging happens before pricing, keyed on (damage class, component), because
    "a panel is bought once" and "a panel is painted once" are facts about the
    vehicle and would be broken by pricing each photograph on its own.

    Where a class/component pair appears in more than one view, the merged group
    takes the worst severity but the **maximum** instance count and area, not the
    sum. See the module docstring: two photographs of one scratch must not cost
    twice as much as one, and this system cannot tell that case apart from two
    genuinely separate scratches.
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
    span several photographs while still charging for each component once.
    """
    seg = SEGMENTS.get(segment) or SEGMENTS[DEFAULT_SEGMENT]
    notes: list[str] = []

    if not pairs:
        return Quotation(
            panel=panel, segment=segment, panels=panels,
            # Empty, not "minor". Nothing was detected, so there is nothing to
            # grade, and "minor" would be a claim the measurement does not
            # support - the results page and the PDF would both print a severity
            # badge for a car the model found undamaged.
            overall_severity="",
            notes=["No damage detected, so no repair cost is estimated."]
                  + list(extra_notes),
        )

    # Pass 1: work out, per component, the largest replacement fraction any
    # single action calls for, and whether anything paints it. This is what
    # stops us buying the same bumper twice.
    max_replace: dict[str, float] = {}
    paints: dict[str, bool] = {}
    for component, group in pairs:
        action = REPAIR_ACTIONS.get((group.class_name, group.severity))
        if action is None:
            notes.append(
                f"No repair action defined for {group.class_name} / "
                f"{group.severity}; skipped."
            )
            continue
        max_replace[component] = max(max_replace.get(component, 0.0),
                                     action.replace_fraction)
        paints[component] = paints.get(component, False) or action.paints

    # Pass 2: build the lines. Part and paint cost are attributed to the line
    # that justified them, and only once per component, so the totals add up to
    # the sum of the lines - a report whose lines do not sum to its total is
    # worse than no report.
    part_charged: set[str] = set()
    paint_charged: set[str] = set()
    lines: list[CostLine] = []

    def replace_fraction_of(pair: tuple[str, DamageGroup]) -> float:
        action = REPAIR_ACTIONS.get((pair[1].class_name, pair[1].severity))
        return action.replace_fraction if action else -1.0

    # Most expensive first, so the line that justifies buying the part is the
    # line charged for it, and the reader sees the bill's driver at the top.
    # Python's sort is stable, so ties keep the order they arrived in.
    ordered = sorted(pairs, key=replace_fraction_of, reverse=True)

    for component, group in ordered:
        action = REPAIR_ACTIONS.get((group.class_name, group.severity))
        if action is None:
            continue
        comp = PANELS.get(component) or PANELS[DEFAULT_PANEL]

        part_cost = 0.0
        if component not in part_charged and max_replace.get(component, 0.0) > 0:
            part_cost = comp.part_cost * seg.part_multiplier * max_replace[component]
            part_charged.add(component)

        paint_cost = 0.0
        if (component not in paint_charged and paints.get(component)
                and comp.paintable):
            paint_cost = comp.paint_cost * seg.part_multiplier
            paint_charged.add(component)

        hours = action.labour_hours * instance_factor(group.instances)
        labour_cost = seg.labour_rate * hours

        lines.append(CostLine(
            class_name=group.class_name,
            class_label=CLASS_LABELS.get(group.class_name, group.class_name),
            severity=group.severity,
            severity_label=SEVERITY_LABELS.get(group.severity, group.severity),
            component=component,
            component_label=comp.label,
            action=action.description,
            instances=group.instances,
            part_cost=round(part_cost, 2),
            labour_cost=round(labour_cost, 2),
            labour_hours=round(hours, 2),
            paint_cost=round(paint_cost, 2),
        ))

    parts_total = round(sum(l.part_cost for l in lines), 2)
    labour_total = round(sum(l.labour_cost for l in lines), 2)
    paint_total = round(sum(l.paint_cost for l in lines), 2)
    subtotal = round(parts_total + labour_total + paint_total, 2)
    gst = round(subtotal * GST_RATE, 2)

    for component in sorted(part_charged):
        if max_replace[component] < 1.0:
            notes.append(
                f"{PANELS[component].label}: repaired rather than replaced, so "
                f"only {max_replace[component]:.0%} of the part price is charged."
            )

    if any(l.instances > 1 for l in lines):
        notes.append(
            f"Repeated damage of the same type is charged at "
            f"{INSTANCE_FACTOR:.0%} labour for each instance after the first, "
            f"because preparation and paint-booth time are shared."
        )

    return Quotation(
        lines=lines,
        panel=panel,
        segment=segment,
        panels=panels,
        parts_total=parts_total,
        labour_total=labour_total,
        paint_total=paint_total,
        subtotal=subtotal,
        gst=gst,
        total=round(subtotal + gst, 2),
        overall_severity=worst(g.severity for _, g in pairs),
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
