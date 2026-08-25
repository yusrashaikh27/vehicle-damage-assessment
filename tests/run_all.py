#!/usr/bin/env python3
"""Run every check in tests/ and report one verdict.

    ./venv/bin/python tests/run_all.py

Each check is a separate process rather than an imported function, for two
reasons. It keeps them independently runnable - a failing check can be re-run on
its own with its full output, which is what you actually want when something
breaks - and it means one check crashing cannot take the runner down with it.

Exit status is 0 only if every check passed, so this is usable as a pre-commit
hook or a CI step without any wrapping.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (script, arguments, what a pass actually tells you)
CHECKS: list[tuple[str, list[str], str]] = [
    ("check_models.py", ["--selftest"],
     "the attribute checker is itself working"),
    ("cost_baseline.py", ["check"],
     "155 recorded quotations are unchanged"),
    ("test_multiview.py", [],
     "the multi-photograph merge obeys its four properties"),
    ("check_models.py", [],
     "every model attribute the admin, API and templates use exists"),
    ("check_templates.py", [],
     "every template tag, filter, URL name and variable resolves"),
]

# The self-test runs first on purpose. If the checker is broken, a clean report
# from it means nothing, so there is no point reading the rest.


def main() -> int:
    width = max(len(name) + len(" ".join(args)) for name, args, _ in CHECKS) + 2
    failed: list[str] = []

    for script, args, claim in CHECKS:
        label = f"{script} {' '.join(args)}".strip()
        proc = subprocess.run(
            [sys.executable, str(HERE / script), *args],
            capture_output=True, text=True,
        )
        ok = proc.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'}  {label:<{width}} {claim}")
        if not ok:
            failed.append(label)
            # Full output only for failures - a passing run should be five lines,
            # readable at a glance, or nobody will run it.
            body = (proc.stdout + proc.stderr).rstrip()
            print("\n".join(f"        {line}" for line in body.splitlines()))
            print()

    print()
    if failed:
        print(f"{len(failed)} of {len(CHECKS)} checks failed: {', '.join(failed)}")
        print()
        print("If a cost_baseline failure is an intended pricing change, read the")
        print("diff above, confirm every moved figure is one you meant to move,")
        print("then re-record it:  python tests/cost_baseline.py write")
        return 1

    print(f"all {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
