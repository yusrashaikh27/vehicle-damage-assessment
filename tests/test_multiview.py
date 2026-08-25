"""Does multi-view pricing actually do what the docstring claims?

Four properties, each one a thing an examiner could reasonably poke at:

1. A single view through estimate_combined equals estimate on the same input.
2. Two photographs of the SAME panel buy that panel once, not twice.
3. Two photographs of DIFFERENT panels buy both panels.
4. Merging never increases the bill relative to the worst single view.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The project root is found by walking up from this file, never hardcoded and
# never assumed to be the working directory - so `python tests/test_multiview.py` works
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

from core.cost import DamageGroup, estimate, estimate_combined  # noqa: E402

failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    print(("pass  " if ok else "FAIL  ") + label + (f"   {detail}" if detail else ""))
    if not ok:
        failures += 1


crack = DamageGroup("crack", "severe", instances=1, area_fraction=0.06,
                    confidence=0.9)
dent = DamageGroup("dent", "moderate", instances=2, area_fraction=0.03,
                   confidence=0.8)
scratch = DamageGroup("scratch", "minor", instances=1, area_fraction=0.004,
                      confidence=0.6)

# ---------------------------------------------------------------- property 1
single = estimate([crack, dent], panel="front_bumper", segment="sedan")
combined_one = estimate_combined([("front_bumper", [crack, dent])],
                                 segment="sedan")
check("one view == estimate()",
      (single.total == combined_one.total
       and single.parts_total == combined_one.parts_total
       and single.labour_total == combined_one.labour_total
       and single.paint_total == combined_one.paint_total
       and len(single.lines) == len(combined_one.lines)),
      f"{single.total} vs {combined_one.total}")

# ---------------------------------------------------------------- property 2
# The same bumper photographed from two angles.
same_panel = estimate_combined(
    [("front_bumper", [crack]), ("front_bumper", [crack])], segment="sedan")
one_shot = estimate_combined([("front_bumper", [crack])], segment="sedan")

bumper_part_lines = [l for l in same_panel.lines if l.part_cost > 0]
check("same panel twice -> one part charge",
      len(bumper_part_lines) == 1,
      f"{len(bumper_part_lines)} line(s) carry a part cost")
check("same panel twice -> identical total to one shot",
      same_panel.total == one_shot.total,
      f"two views {same_panel.total} vs one view {one_shot.total}")
check("same panel twice -> a note explains the merge",
      any("more than one photograph" in n for n in same_panel.notes))

# ---------------------------------------------------------------- property 3
two_panels = estimate_combined(
    [("front_bumper", [crack]), ("rear_bumper", [crack])], segment="sedan")
components = {l.component for l in two_panels.lines if l.part_cost > 0}
check("different panels -> both parts charged",
      components == {"front_bumper", "rear_bumper"},
      str(sorted(components)))
check("different panels -> dearer than one panel",
      two_panels.total > one_shot.total,
      f"{two_panels.total} > {one_shot.total}")

# ---------------------------------------------------------------- property 4
# Four overlapping views of one corner, each finding the same three damages.
noisy = estimate_combined(
    [("front_bumper", [crack, dent, scratch])] * 4, segment="suv")
clean = estimate_combined([("front_bumper", [crack, dent, scratch])],
                          segment="suv")
check("four overlapping views cost the same as one",
      noisy.total == clean.total,
      f"{noisy.total} vs {clean.total}")

# ------------------------------------------------------- lines sum to totals
for label, q in (("one view", combined_one), ("two panels", two_panels),
                 ("noisy", noisy)):
    line_sum = round(sum(l.subtotal for l in q.lines), 2)
    check(f"lines sum to subtotal ({label})",
          abs(line_sum - q.subtotal) < 0.011,
          f"lines {line_sum} vs subtotal {q.subtotal}")

# --------------------------------------------------------- nothing found
empty = estimate_combined([("front_bumper", []), ("rear_bumper", [])],
                          segment="hatchback")
check("no damage in any view -> zero and a note",
      empty.total == 0.0 and empty.notes and not empty.lines,
      str(empty.notes))

# --------------------------------------------- panels recorded on the result
check("panels tuple records every view",
      two_panels.panels == ("front_bumper", "rear_bumper"),
      str(two_panels.panels))
check("panel collapses to the shared one when views agree",
      same_panel.panel == "front_bumper", same_panel.panel)
check("panel is 'other' when views disagree",
      two_panels.panel == "other", two_panels.panel)

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
