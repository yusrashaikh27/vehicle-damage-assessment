"""Single source of truth for the damage taxonomy, severity rules and cost rates.

Everything the system claims about a car is derived from the tables in this file.
That is the point of putting them all here: in a viva you can open one file and
show exactly where a number came from, instead of hunting through view code.

Three things live here:

1. CLASS_NAMES     - the 7 damage classes, in the same order as dataset/data.yaml.
2. SEVERITY_RULES  - how a mask area becomes Minor / Moderate / Severe.
3. the cost tables - vehicle segments, panels (with a low-high repair price from a
                     published price list) and the repair action implied by each
                     (class, severity) pair.

The cost model is REPAIR ONLY and every price is a RANGE. The only exceptions are
lamps and glass, which cannot be repaired and are priced from the list's
replacement table, and a flat tyre, which the list does not cover at all.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 1. Damage classes
# ---------------------------------------------------------------------------
# Order matters: these are the class ids the trained model outputs, and they
# must match dataset/data.yaml, which tools/prepare_dataset.py generates from
# its MERGE_GROUPS list. If you change the taxonomy, change it there and
# regenerate - do not hand-edit the order here.

CLASS_NAMES: list[str] = [
    "scratch",          # 0
    "dent",             # 1
    "crack",            # 2
    "lamp_broken",      # 3
    "dislocated_part",  # 4
    "glass_shatter",    # 5
    "tire_flat",        # 6
]

# Shown to the user; the model's names are for us, not for them.
CLASS_LABELS: dict[str, str] = {
    "scratch": "Scratch",
    "dent": "Dent",
    "crack": "Crack",
    "lamp_broken": "Broken lamp",
    "dislocated_part": "Dislocated / missing part",
    "glass_shatter": "Shattered glass",
    "tire_flat": "Flat tyre",
}

SEVERITY_ORDER: list[str] = ["minor", "moderate", "severe"]

SEVERITY_LABELS: dict[str, str] = {
    "minor": "Minor",
    "moderate": "Moderate",
    "severe": "Severe",
}


# ---------------------------------------------------------------------------
# 2. Severity
# ---------------------------------------------------------------------------
# The dataset has no severity labels, so severity CANNOT be learned from it.
# It is computed from the segmentation mask area, expressed as a fraction of a
# reference area (see core/severity.py for what the reference is and why).
#
# Two knobs per class, because area alone is not enough:
#
#   minor_max / moderate_max  - the area fractions where the bands change.
#                               Thin damage (a crack) covers less area than
#                               broad damage (a dent) for the same seriousness,
#                               so the thresholds differ by class.
#
#   floor                     - the least severe verdict this class can ever
#                               receive. A shattered windscreen is not "minor"
#                               however small the shattered patch looks, because
#                               the repair action is the same either way: the
#                               glass gets replaced. Same for a flat tyre and a
#                               broken lamp. This is where domain knowledge
#                               enters the rule, and it is the part worth
#                               defending.


@dataclass(frozen=True)
class SeverityRule:
    minor_max: float
    moderate_max: float
    floor: str
    note: str


SEVERITY_RULES: dict[str, SeverityRule] = {
    "scratch": SeverityRule(
        minor_max=0.015, moderate_max=0.060, floor="minor",
        note="Area scales with how much of the panel needs re-painting.",
    ),
    "dent": SeverityRule(
        minor_max=0.010, moderate_max=0.040, floor="minor",
        note="A large dent implies panel-beating, not just filler and paint.",
    ),
    "crack": SeverityRule(
        minor_max=0.008, moderate_max=0.030, floor="minor",
        note="Cracks are thin, so they cover little area - lower thresholds.",
    ),
    "lamp_broken": SeverityRule(
        minor_max=0.0, moderate_max=0.025, floor="moderate",
        note="A cracked lens and a smashed lamp are both replaced; never minor.",
    ),
    "dislocated_part": SeverityRule(
        minor_max=0.0, moderate_max=0.050, floor="moderate",
        note="Refit at moderate; a large exposed area means the panel is gone.",
    ),
    "glass_shatter": SeverityRule(
        minor_max=0.0, moderate_max=0.040, floor="moderate",
        note="Glass is always replaced, so the floor is moderate.",
    ),
    "tire_flat": SeverityRule(
        minor_max=0.0, moderate_max=0.030, floor="moderate",
        note="Moderate = puncture repair, severe = the tyre is replaced.",
    ),
}


# ---------------------------------------------------------------------------
# 3. Cost model
# ---------------------------------------------------------------------------
# Every price is a RANGE (low, high) taken from a published repair price list.
#
#   line range  = panel range  x  where the severity sits inside it
#                              x  segment multiplier
#                              x  instance factor (repeated damage of one type)
#
# A panel with several damage types is prepared and painted once, so the dearest
# damage is charged in full and the rest at INSTANCE_FACTOR (see core/cost.py).
# GST is then added to both ends of the range.

CURRENCY = "INR"
CURRENCY_SYMBOL = "₹"

GST_RATE = 0.18  # Motor-vehicle repair in India attracts 18% GST on parts and labour.

# Printed in the PDF, the results page and the notes. FILL IN the page URL and the
# date you read it before submission. The list says "before offers" and does not
# say whether GST is included; this project treats the prices as EXCLUDING GST
# and adds it. If your guide wants them taken as GST-inclusive, set GST_RATE to
# 0.0 and change this sentence.
COST_SOURCE = (
    "GoMechanic car body repair price list: approximate per-panel price ranges "
    "for common passenger cars, before offers [add page URL and date accessed]. "
    "Prices are treated as excluding GST."
)

# The presented total is rounded outward to a round figure at the scale of the
# estimate: Rs. 500 steps for small totals, Rs. 1,000 from Rs. 20,000 up. The low
# end is rounded down and the high end up, so rounding only ever widens a range.
# Ascending by threshold; the first threshold must be 0.
RANGE_STEP_TIERS: tuple[tuple[float, int], ...] = (
    (0.0, 500),
    (20_000.0, 1_000),
)

# Each cost line is rounded outward to this step (Rs.), so a line never reads as
# falsely precise. The subtotal is the sum of the rounded lines, so the lines
# always add up to it.
RANGE_LINE_STEP = 100


@dataclass(frozen=True)
class Segment:
    label: str
    # Applied to BOTH ends of every price. The price list is for "common
    # passenger cars" (the hatchback baseline); larger and more premium cars
    # cost more. These multipliers are a JUDGEMENT, not from the list - say so.
    price_multiplier: float


SEGMENTS: dict[str, Segment] = {
    "hatchback": Segment("Hatchback", 1.00),
    "sedan":     Segment("Sedan",     1.25),
    "suv":       Segment("SUV / MUV", 1.60),
    "luxury":    Segment("Luxury / imported", 3.00),
}

DEFAULT_SEGMENT = "hatchback"


@dataclass(frozen=True)
class Panel:
    label: str
    # Repair price range for the whole entry, Rs., hatchback baseline: denting,
    # filling and painting, all-in. 0 means "this component is not repaired".
    repair_low: float = 0.0
    repair_high: float = 0.0
    # Replacement price range, Rs. Only set for components that cannot be
    # repaired (lamps and glass). 0 means "never replaced in this project".
    replace_low: float = 0.0
    replace_high: float = 0.0
    # How many physical panels this entry stands for. The price range is already
    # the total for the whole entry; this is used for flat per-panel prices
    # (e.g. a minor-scratch touch-up on each of two doors).
    panel_count: int = 1


# The panel is supplied by the user at upload, not predicted. No class in the
# dataset names a component - the labels are damage *types* - so the component
# has to come from somewhere, and asking is both honest and accurate.
#
# Ranges are from the price list's "denting and painting" table unless noted.
PANELS: dict[str, Panel] = {
    "front_bumper":  Panel("Front bumper",          1500, 3500),
    "rear_bumper":   Panel("Rear bumper",           1500, 3500),
    "bonnet":        Panel("Bonnet / hood",         2000, 4500),
    "boot":          Panel("Boot / tailgate",       2000, 4000),
    "roof":          Panel("Roof",                  3000, 7000),
    "front_fender":  Panel("Front fender",          1500, 3500),
    "rear_fender":   Panel("Rear fender / quarter", 2000, 4500),  # list: quarter panel
    "front_door":    Panel("Front door",            2000, 4000),
    "rear_door":     Panel("Rear door",             2000, 4000),
    # Front + rear door on the same side: the two doors' ranges added together
    # (2,000-4,000 + 2,000-4,000). Needed because one photograph often shows both
    # doors, and a single-panel choice would price two damaged doors as one.
    "doors_same_side": Panel("Front + rear door (same side)", 4000, 8000,
                             panel_count=2),
    # Glass and lamps cannot be repaired: replacement ranges from the list's
    # "windshield, glass and lighting" table. The only non-repair prices here.
    "windscreen":    Panel("Windscreen",   replace_low=3000, replace_high=12000),
    "rear_glass":    Panel("Rear glass",   replace_low=3000, replace_high=10000),
    "door_glass":    Panel("Door glass",   replace_low=1500, replace_high=5000),
    "headlamp":      Panel("Headlamp",     replace_low=2000, replace_high=12000),
    "taillamp":      Panel("Taillamp",     replace_low=1000, replace_high=6000),
    # NOT directly in the list. The grille uses the list's "small paint touch-up"
    # range as a stand-in. Replace with a cited figure if you find one.
    "grille":        Panel("Front grille",   500, 1500),
    # List: "alloy wheel - painting, per wheel".
    "wheel":         Panel("Wheel / tyre",  1000, 2000),
    # NOT in the list: a generic mid-range body-panel price, used when the owner
    # does not say which panel. Replace with a cited figure if you find one.
    "other":         Panel("Other / unspecified", 2000, 4000),
}

DEFAULT_PANEL = "other"


@dataclass(frozen=True)
class Angle:
    label: str
    default_panel: str  # the component this view usually shows
    hint: str           # what a usable photograph from this angle looks like


# The walk-around. A damage assessor photographs a vehicle from a fixed set of
# positions so that nothing is missed and two assessments of the same car are
# comparable; the same convention is used here.
#
# Each angle carries the component it *usually* shows, which pre-selects the panel
# dropdown. It is a default, not a decision: a side-on photograph might be of the
# rear door rather than the front, and the owner can say so. The panel matters
# because it sets the price, so a wrong default that goes uncorrected costs real
# money - hence the defaults are conservative and every one is editable.
#
# The four corners are 3/4 views. They are the most useful shots on a damaged car
# because they show two faces at once, which is why insurers ask for them.
ANGLES: dict[str, Angle] = {
    "front":       Angle("Front",             "front_bumper",
                         "Square to the front, whole bumper and grille in frame."),
    "front_left":  Angle("Front left (3/4)",  "front_fender",
                         "From the front-left corner, so front and left both show."),
    "left":        Angle("Left side",         "doors_same_side",
                         "Square to the side, full length of the car in frame."),
    "rear_left":   Angle("Rear left (3/4)",   "rear_fender",
                         "From the rear-left corner, showing rear and left."),
    "rear":        Angle("Rear",              "rear_bumper",
                         "Square to the back, whole bumper and tailgate."),
    "rear_right":  Angle("Rear right (3/4)",  "rear_fender",
                         "From the rear-right corner, showing rear and right."),
    "right":       Angle("Right side",        "doors_same_side",
                         "Square to the side, full length of the car in frame."),
    "front_right": Angle("Front right (3/4)", "front_fender",
                         "From the front-right corner, showing front and right."),
    "roof":        Angle("Roof",              "roof",
                         "Only if the roof or windscreen is damaged."),
    "closeup":     Angle("Close-up of damage", "other",
                         "Fill the frame with the damage itself. Best single shot "
                         "for accuracy, since area is measured against what is "
                         "visible."),
}

DEFAULT_ANGLE = "closeup"


# How a (class, severity) pair is priced. Four modes:
#
#   "repair"   - the panel's repair range, taking the slice from low_share to
#                high_share of it (0.0 = bottom of the range, 1.0 = top). Minor
#                damage sits in the lower part, severe in the upper part. These
#                slices are a JUDGEMENT, not a measurement.
#   "flat"     - a fixed per-panel range (the list's minor-scratch / touch-up
#                price), independent of the panel's own range.
#   "replace"  - the component's replacement range, for things that cannot be
#                repaired (lamps, glass). One part is bought however many
#                instances there are.
#   "unpriced" - the price list has no figure, so nothing is charged and the
#                report says so.
REPAIR_MODES = ("repair", "flat", "replace", "unpriced")


@dataclass(frozen=True)
class RepairAction:
    description: str
    mode: str = "repair"
    low_share: float = 0.0
    high_share: float = 1.0
    flat_low: float = 0.0
    flat_high: float = 0.0


# (class, severity) -> what the workshop actually does, repair only.
REPAIR_ACTIONS: dict[tuple[str, str], RepairAction] = {
    # Scratch: minor uses the list's minor-scratch / touch-up price (Rs. 500-1,500);
    # worse scratches need the panel re-painted, so use the panel range.
    ("scratch", "minor"):    RepairAction("Polish and spot re-paint", mode="flat",
                                          flat_low=500, flat_high=1500),
    ("scratch", "moderate"): RepairAction("Sand, prime and re-paint panel",
                                          low_share=0.25, high_share=0.75),
    ("scratch", "severe"):   RepairAction("Full panel re-paint and blend",
                                          low_share=0.50, high_share=1.00),

    # Dent: the list's "denting and painting" range.
    ("dent", "minor"):       RepairAction("Dent removal and spot re-paint",
                                          low_share=0.00, high_share=0.50),
    ("dent", "moderate"):    RepairAction("Pull out, fill and re-paint",
                                          low_share=0.25, high_share=0.75),
    ("dent", "severe"):      RepairAction("Panel beating, filling and re-paint",
                                          low_share=0.50, high_share=1.00),

    # Crack: the list has no crack-repair price, so cracks use the same panel
    # range as dents but one band higher (more work than a dent of the same
    # severity). ASSUMPTION - state it in the report.
    ("crack", "minor"):      RepairAction("Weld or fill crack and touch-up",
                                          low_share=0.25, high_share=0.75),
    ("crack", "moderate"):   RepairAction("Section, bond and re-paint",
                                          low_share=0.50, high_share=1.00),
    ("crack", "severe"):     RepairAction("Crack repair, filling and full re-paint",
                                          low_share=0.50, high_share=1.00),

    # Lamps and glass cannot be repaired: replacement range of the component.
    ("lamp_broken", "moderate"): RepairAction("Replace lamp assembly", mode="replace"),
    ("lamp_broken", "severe"):   RepairAction("Replace lamp assembly", mode="replace"),

    # Dislocated part: project is repair-only, so it is refitted / straightened and
    # priced from the panel's repair range. ASSUMPTION - not a listed service.
    ("dislocated_part", "moderate"): RepairAction("Refit, realign and touch-up",
                                                  low_share=0.25, high_share=0.75),
    ("dislocated_part", "severe"):   RepairAction("Straighten, refit and re-paint panel",
                                                  low_share=0.50, high_share=1.00),

    ("glass_shatter", "moderate"): RepairAction("Replace glass", mode="replace"),
    ("glass_shatter", "severe"):   RepairAction("Replace glass", mode="replace"),

    # Not in the price list at all, so not charged. The report says so.
    ("tire_flat", "moderate"): RepairAction("Puncture repair (not in price list)",
                                            mode="unpriced"),
    ("tire_flat", "severe"):   RepairAction("Tyre replacement (not in price list)",
                                            mode="unpriced"),
}


# ---------------------------------------------------------------------------
# Consistency checks - run at import so a bad edit fails immediately and
# loudly, rather than silently producing a wrong quotation.
# ---------------------------------------------------------------------------

def _validate() -> None:
    assert len(CLASS_NAMES) == len(set(CLASS_NAMES)), "duplicate class name"
    assert set(CLASS_LABELS) == set(CLASS_NAMES), "CLASS_LABELS out of sync"
    assert set(SEVERITY_RULES) == set(CLASS_NAMES), "SEVERITY_RULES out of sync"

    for name, rule in SEVERITY_RULES.items():
        assert rule.floor in SEVERITY_ORDER, f"{name}: bad floor {rule.floor!r}"
        assert 0.0 <= rule.minor_max <= rule.moderate_max, f"{name}: bands out of order"

    # Every band a class can actually reach must have a repair action, and no
    # action may exist for a band the class can never reach (that would be dead
    # code pretending to be a rule).
    for name, rule in SEVERITY_RULES.items():
        floor_index = SEVERITY_ORDER.index(rule.floor)
        reachable = set(SEVERITY_ORDER[floor_index:])
        defined = {sev for (cls, sev) in REPAIR_ACTIONS if cls == name}
        assert defined == reachable, (
            f"{name}: repair actions {sorted(defined)} != reachable severities "
            f"{sorted(reachable)}"
        )

    for key, action in REPAIR_ACTIONS.items():
        assert action.mode in REPAIR_MODES, f"{key}: unknown mode {action.mode!r}"
        assert 0.0 <= action.low_share <= action.high_share <= 1.0, (
            f"{key}: shares must satisfy 0 <= low <= high <= 1"
        )
        if action.mode == "flat":
            assert 0 < action.flat_low <= action.flat_high, (
                f"{key}: a flat price needs 0 < flat_low <= flat_high"
            )

    assert DEFAULT_PANEL in PANELS
    assert DEFAULT_SEGMENT in SEGMENTS

    for key, panel in PANELS.items():
        assert panel.panel_count >= 1, f"panel {key!r}: panel_count must be at least 1"
        assert 0 <= panel.repair_low <= panel.repair_high, (
            f"panel {key!r}: repair range out of order"
        )
        assert 0 <= panel.replace_low <= panel.replace_high, (
            f"panel {key!r}: replace range out of order"
        )

    for key, seg in SEGMENTS.items():
        assert seg.price_multiplier > 0, f"segment {key!r}: multiplier must be positive"

    assert 0.0 <= GST_RATE < 1.0, "GST_RATE must be a fraction"
    assert RANGE_LINE_STEP > 0, "RANGE_LINE_STEP must be positive"

    # The rounding tiers must start at zero, ascend, and use positive steps.
    thresholds = [t for t, _ in RANGE_STEP_TIERS]
    assert thresholds[0] == 0.0, "RANGE_STEP_TIERS must start at threshold 0"
    assert thresholds == sorted(thresholds), "RANGE_STEP_TIERS must ascend"
    assert all(step > 0 for _, step in RANGE_STEP_TIERS), "steps must be positive"

    # Every angle must default to a panel that actually has a price, or the first
    # upload from that angle would be priced against a missing rate card entry.
    assert DEFAULT_ANGLE in ANGLES
    for key, angle in ANGLES.items():
        assert angle.default_panel in PANELS, (
            f"angle {key!r}: default_panel {angle.default_panel!r} not in PANELS"
        )


_validate()


# Convenience forms for Django's ChoiceField / model choices, so the form and
# the cost model can never drift apart.
PANEL_CHOICES = [(key, panel.label) for key, panel in PANELS.items()]
SEGMENT_CHOICES = [(key, seg.label) for key, seg in SEGMENTS.items()]
CLASS_CHOICES = [(name, CLASS_LABELS[name]) for name in CLASS_NAMES]
SEVERITY_CHOICES = [(s, SEVERITY_LABELS[s]) for s in SEVERITY_ORDER]
ANGLE_CHOICES = [(key, angle.label) for key, angle in ANGLES.items()]