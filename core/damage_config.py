"""Single source of truth for the damage taxonomy, severity rules and cost rates.

Everything the system claims about a car is derived from the tables in this file.
That is the point of putting them all here: in a viva you can open one file and
show exactly where a number came from, instead of hunting through view code.

Three things live here:

1. CLASS_NAMES     - the 7 damage classes, in the same order as dataset/data.yaml.
2. SEVERITY_RULES  - how a mask area becomes Minor / Moderate / Severe.
3. the cost tables - vehicle segments, panels, and the repair action implied by
                     each (class, severity) pair.

IMPORTANT, and say this out loud rather than being caught on it: the money
figures below are *indicative*. They were set from typical Indian
authorised-workshop pricing to make the system demonstrable end to end. Before
the final submission, replace them with figures you can cite - an insurer's
schedule of rates, a dealership quotation, or a published survey - and note the
source in COST_SOURCE. A cost model whose provenance you can state is worth far
more marks than one with suspiciously precise numbers and no origin.
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
# cost = part_cost(panel) x segment_multiplier x replace_fraction(class, severity)
#      + labour_rate(segment) x labour_hours(class, severity)
#      + paint_material(panel) x segment_multiplier   [only if the repair paints]
#      then GST on the total.
#
# The classes were merged by repair action in prepare_dataset.py precisely so
# that this table can exist: each class maps onto one thing a workshop does.

CURRENCY = "INR"
CURRENCY_SYMBOL = "₹"

GST_RATE = 0.18  # Motor-vehicle repair in India attracts 18% GST on parts and labour.

COST_SOURCE = (
    "Indicative rates set from typical Indian authorised-workshop pricing "
    "(2025). NOT yet sourced from a citable schedule of rates - replace before "
    "final submission and record the source here."
)


@dataclass(frozen=True)
class Segment:
    label: str
    part_multiplier: float   # applied to the hatchback baseline part cost
    labour_rate: float       # INR per hour


# Larger and more premium cars cost more in both parts and labour. One
# multiplier per segment keeps the panel table to a single column.
SEGMENTS: dict[str, Segment] = {
    "hatchback": Segment("Hatchback", 1.00, 450.0),
    "sedan":     Segment("Sedan",     1.25, 550.0),
    "suv":       Segment("SUV / MUV", 1.60, 700.0),
    "luxury":    Segment("Luxury / imported", 3.00, 1400.0),
}

DEFAULT_SEGMENT = "hatchback"


@dataclass(frozen=True)
class Panel:
    label: str
    part_cost: float    # hatchback baseline, INR, for a new part
    paint_cost: float   # material only (primer, base, clear); labour is separate
    paintable: bool = True


# The panel is supplied by the user at upload, not predicted. No class in the
# dataset names a component - the labels are damage *types* - so the component
# has to come from somewhere, and asking is both honest and accurate. The report
# already says the user "uploads an image along with vehicle details".
PANELS: dict[str, Panel] = {
    "front_bumper":  Panel("Front bumper",        4500, 2500),
    "rear_bumper":   Panel("Rear bumper",         4200, 2500),
    "bonnet":        Panel("Bonnet / hood",       7500, 3200),
    "boot":          Panel("Boot / tailgate",     7800, 3000),
    "roof":          Panel("Roof",               11000, 4000),
    "front_fender":  Panel("Front fender",        3800, 2000),
    "rear_fender":   Panel("Rear fender / quarter", 5200, 2400),
    "front_door":    Panel("Front door",          9500, 2800),
    "rear_door":     Panel("Rear door",           9000, 2800),
    "windscreen":    Panel("Windscreen",          9000,    0, paintable=False),
    "rear_glass":    Panel("Rear glass",          6500,    0, paintable=False),
    "door_glass":    Panel("Door glass",          3500,    0, paintable=False),
    "headlamp":      Panel("Headlamp",            6500,    0, paintable=False),
    "taillamp":      Panel("Taillamp",            3500,    0, paintable=False),
    "grille":        Panel("Front grille",        3200, 1500),
    "wheel":         Panel("Wheel / tyre",        4500,    0, paintable=False),
    "other":         Panel("Other / unspecified", 5000, 2500),
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
# because it sets the part price, so a wrong default that goes uncorrected costs
# real money - hence the defaults are conservative and every one is editable.
#
# The four corners are 3/4 views. They are the most useful shots on a damaged car
# because they show two faces at once, which is why insurers ask for them.
ANGLES: dict[str, Angle] = {
    "front":       Angle("Front",             "front_bumper",
                         "Square to the front, whole bumper and grille in frame."),
    "front_left":  Angle("Front left (3/4)",  "front_fender",
                         "From the front-left corner, so front and left both show."),
    "left":        Angle("Left side",         "front_door",
                         "Square to the side, full length of the car in frame."),
    "rear_left":   Angle("Rear left (3/4)",   "rear_fender",
                         "From the rear-left corner, showing rear and left."),
    "rear":        Angle("Rear",              "rear_bumper",
                         "Square to the back, whole bumper and tailgate."),
    "rear_right":  Angle("Rear right (3/4)",  "rear_fender",
                         "From the rear-right corner, showing rear and right."),
    "right":       Angle("Right side",        "front_door",
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


@dataclass(frozen=True)
class RepairAction:
    description: str
    replace_fraction: float  # 0.0 = repair in place, 1.0 = fit a new part
    labour_hours: float
    paints: bool


# (class, severity) -> what the workshop actually does.
# replace_fraction is a fraction rather than a boolean because some repairs sit
# in between: a moderately cracked bumper is often sectioned and bonded rather
# than replaced, which costs roughly a third of a new one in consumables.
REPAIR_ACTIONS: dict[tuple[str, str], RepairAction] = {
    ("scratch", "minor"):    RepairAction("Polish and spot re-paint", 0.00, 1.0, True),
    ("scratch", "moderate"): RepairAction("Sand, prime and re-paint panel", 0.00, 2.0, True),
    ("scratch", "severe"):   RepairAction("Full panel re-paint and blend", 0.00, 3.5, True),

    ("dent", "minor"):       RepairAction("Paintless dent removal", 0.00, 1.5, False),
    ("dent", "moderate"):    RepairAction("Pull out, fill and re-paint", 0.00, 3.0, True),
    ("dent", "severe"):      RepairAction("Panel beating, filling and re-paint", 0.00, 5.0, True),

    ("crack", "minor"):      RepairAction("Plastic weld and touch-up", 0.00, 1.5, True),
    ("crack", "moderate"):   RepairAction("Section, bond and re-paint", 0.35, 3.0, True),
    ("crack", "severe"):     RepairAction("Replace part and re-paint", 1.00, 4.0, True),

    ("lamp_broken", "moderate"): RepairAction("Replace lamp assembly", 1.00, 1.0, False),
    ("lamp_broken", "severe"):   RepairAction("Replace lamp assembly and mounting", 1.00, 1.5, False),

    ("dislocated_part", "moderate"): RepairAction("Refit and realign panel", 0.00, 2.5, False),
    ("dislocated_part", "severe"):   RepairAction("Replace and refit panel", 1.00, 4.5, True),

    ("glass_shatter", "moderate"): RepairAction("Replace glass", 1.00, 2.0, False),
    ("glass_shatter", "severe"):   RepairAction("Replace glass and clear debris", 1.00, 2.5, False),

    ("tire_flat", "moderate"): RepairAction("Puncture repair and balancing", 0.00, 0.5, False),
    ("tire_flat", "severe"):   RepairAction("Replace tyre and balance", 1.00, 0.75, False),
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
        assert 0.0 <= action.replace_fraction <= 1.0, f"{key}: bad replace_fraction"
        assert action.labour_hours > 0, f"{key}: labour_hours must be positive"

    assert DEFAULT_PANEL in PANELS
    assert DEFAULT_SEGMENT in SEGMENTS

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
