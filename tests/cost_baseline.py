"""Capture what core.cost.estimate() returns today, before it is refactored.

The refactor moves the two-pass pricing onto (component, group) pairs so it can
price damage seen across several photographs. The single-panel path must come out
byte-identical. This script writes a baseline; the same script run afterwards
compares against it.

Usage:  python3 cost_baseline.py write    # before the refactor
        python3 cost_baseline.py check    # after
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

# The project root is found by walking up from this file, never hardcoded and
# never assumed to be the working directory - so `python tests/cost_baseline.py` works
# from the project root, from inside tests/, or from anywhere else.
def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "assessment" / "models.py").exists():
            return candidate
    raise SystemExit(
        "Cannot find the project root: no parent of this file contains "
        "assessment/models.py. Run this from inside the project."
    )


ROOT = _project_root()

# core/ is a plain package at the project root; this is what lets these
# tests import it without Django or a settings module.
sys.path.insert(0, str(ROOT))

from core.cost import DamageGroup, estimate  # noqa: E402
from core.damage_config import (  # noqa: E402
    CLASS_NAMES,
    PANELS,
    SEGMENTS,
    SEVERITY_ORDER,
)

# Beside this script, so the recorded numbers travel with the test that
# checks them rather than living in someone's scratch directory.
BASELINE = Path(__file__).resolve().parent / "cost_baseline.json"


def cases():
    """A spread wide enough that a behaviour change cannot hide.

    Includes: every class at every severity on its own; the empty set; several
    classes together on one panel (which exercises buy-the-part-once); and
    repeated instances (which exercises the labour instance factor).
    """
    panels = list(PANELS)
    segments = list(SEGMENTS)

    # 1. one group at a time, every class x severity, on a rotating panel/segment
    for i, (cls, sev) in enumerate(itertools.product(CLASS_NAMES, SEVERITY_ORDER)):
        yield {
            "label": f"single/{cls}/{sev}",
            "groups": [{"class_name": cls, "severity": sev, "instances": 1,
                        "area_fraction": 0.01, "confidence": 0.9}],
            "panel": panels[i % len(panels)],
            "segment": segments[i % len(segments)],
        }

    # 2. nothing found
    for segment in segments:
        yield {"label": f"empty/{segment}", "groups": [],
               "panel": "front_bumper", "segment": segment}

    # 3. several classes on one panel - the part-bought-once path
    combos = [
        ("scratch", "dent"),
        ("crack", "dislocated_part"),
        ("scratch", "dent", "crack"),
        ("lamp_broken", "glass_shatter", "tire_flat"),
        ("dent", "crack", "dislocated_part", "scratch"),
    ]
    for combo, panel, segment in itertools.product(combos, panels[:6], segments):
        yield {
            "label": f"multi/{'+'.join(combo)}/{panel}/{segment}",
            "groups": [{"class_name": c, "severity": "severe", "instances": 1,
                        "area_fraction": 0.05, "confidence": 0.8} for c in combo],
            "panel": panel,
            "segment": segment,
        }

    # 4. repeated instances - the labour instance factor and its cap
    for n in (1, 2, 3, 5, 8, 20):
        yield {
            "label": f"instances/{n}",
            "groups": [{"class_name": "scratch", "severity": "moderate",
                        "instances": n, "area_fraction": 0.02, "confidence": 0.7}],
            "panel": "front_door_left", "segment": "sedan",
        }

    # 5. mixed severities together
    for segment in segments:
        yield {
            "label": f"mixed/{segment}",
            "groups": [
                {"class_name": "scratch", "severity": "minor", "instances": 3,
                 "area_fraction": 0.004, "confidence": 0.6},
                {"class_name": "dent", "severity": "moderate", "instances": 2,
                 "area_fraction": 0.03, "confidence": 0.7},
                {"class_name": "crack", "severity": "severe", "instances": 1,
                 "area_fraction": 0.09, "confidence": 0.9},
            ],
            "panel": "rear_bumper", "segment": segment,
        }


def run() -> dict:
    out = {}
    for case in cases():
        groups = [DamageGroup(**g) for g in case["groups"]]
        q = estimate(groups, panel=case["panel"], segment=case["segment"])
        out[case["label"]] = {
            "lines": [asdict(l) for l in q.lines],
            "panel": q.panel, "segment": q.segment,
            "parts_total": q.parts_total, "labour_total": q.labour_total,
            "paint_total": q.paint_total, "subtotal": q.subtotal,
            "gst": q.gst, "total": q.total,
            "overall_severity": q.overall_severity,
            "currency": q.currency, "notes": q.notes,
        }
    return out


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    current = run()

    if mode == "write":
        BASELINE.write_text(json.dumps(current, indent=1, sort_keys=True))
        print(f"baseline written: {len(current)} cases -> {BASELINE.name}")
        return 0

    expected = json.loads(BASELINE.read_text())
    if set(expected) != set(current):
        print("FAIL: case set changed")
        return 1

    bad = [k for k in expected if expected[k] != current[k]]
    if bad:
        print(f"FAIL: {len(bad)} of {len(expected)} cases changed")
        for k in bad[:5]:
            print(f"\n--- {k}")
            for field_name in expected[k]:
                if expected[k][field_name] != current[k][field_name]:
                    print(f"    {field_name}:")
                    print(f"      was {expected[k][field_name]}")
                    print(f"      now {current[k][field_name]}")
        return 1

    print(f"PASS: all {len(expected)} cases identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
