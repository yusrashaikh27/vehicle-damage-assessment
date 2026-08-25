# Tests

Four checks, none of which need a GPU, a trained model, a database or a browser.
That is deliberate: they are the parts of the system that can be verified by
argument rather than by looking at a screenshot, so they are the parts worth
automating.

Run them all:

    ./venv/bin/python tests/run_all.py

Or individually, from anywhere — each one finds the project root by walking up
from its own file:

| Command | What it proves |
| --- | --- |
| `python tests/cost_baseline.py check` | The cost engine's output has not changed by accident. 155 recorded cases. |
| `python tests/test_multiview.py` | The multi-photograph merge obeys the four properties it claims to. |
| `python tests/check_models.py` | Every model attribute the admin, the API and the templates ask for actually exists. |
| `python tests/check_templates.py` | Every template tag, filter, URL name and context variable resolves. |

## 1. cost_baseline.py — a regression baseline

`cost_baseline.py write` records the full quotation for 155 combinations of
damage class, severity, panel and segment into `cost_baseline.json`.
`cost_baseline.py check` re-prices all 155 and reports any figure that moved.

This is not a correctness test — it cannot be, because there is no ground-truth
repair bill to compare against. It is a *change detector*. The pricing rules
interact (a panel is bought once, painted once, labour adds sublinearly), so it
is entirely possible to edit one rate and silently change forty other
quotations. This catches that.

The discipline that makes it worth having: when a change to the cost engine is
intentional, read the diff, confirm each moved figure is one you meant to move,
then re-record with `write`. A baseline that is permanently red detects nothing.

Worked example — the last time it fired, it reported exactly four changes, all
of them `empty/<segment>`, all on `overall_severity`, from `minor` to empty. That
was the intended fix: a car the model found no damage on was claiming a severity
grade of "minor", which the results page rendered as a badge next to a ₹0 total.
No money moved, so the change was in scope, and the baseline was re-recorded.

## 2. test_multiview.py — properties of the merge

Pricing a walk-around is not pricing each photograph and adding up. A bumper
photographed from the front and again from the front-left is one bumper to buy,
so `estimate_combined` merges views before pricing. These are the properties
that merge has to satisfy:

- **One photograph is unchanged.** A single-view walk-around must price exactly
  as `estimate` would. Multi-view support must not move the single-view answer.
- **Duplicating a photograph changes nothing.** Uploading the same shot twice
  must not double the bill. This is the property the whole design exists for.
- **The worst severity wins.** A dent graded minor in one frame and severe in
  another is a severe dent; the frame that saw it better is the one to trust.
- **The panel is `other` when views disagree.** There is no single panel for a
  walk-around covering three components, and inventing one would be a claim the
  data does not support.

Be candid about the limit these tests document rather than hide: merging takes
the **maximum** per-component figure, not the sum, because this system does not
re-identify damage across views. It cannot tell one dent photographed twice from
two separate dents. Maximum is the conservative reading — it can understate
genuinely separate damage, but it cannot invent damage that is not there.

## 3. check_models.py — the attribute checker

Django resolves template and admin attributes at *render* time, so a typo in
`list_display` or a renamed field referenced by a serializer is invisible until
the page is opened. In a viva that surfaces as a blank admin column or a 500
error mid-demo.

This parses `assessment/models.py` with `ast` and builds the set of real fields,
properties, methods and reverse accessors on each model, then resolves every
attribute path that `admin.py`, `api.py` and the templates ask for.

`ast` rather than importing Django, because it runs with no settings module, no
database and no installed packages.

It has `--selftest`, and that is not decoration. Its first run reported 16
failures, every one of them a false positive, because the test for "is this a
field?" was `name.endswith("Field")` — and `ForeignKey` does not end in "Field".
A checker that is confidently wrong is worse than no checker, so there are now 16
self-test cases, one of them named after that exact bug. Run
`python tests/check_models.py --selftest` before trusting a clean report.

## 4. check_templates.py — the template checker

Same failure mode, same reasoning: template errors are runtime errors. This
resolves every `{% url %}` name against `assessment/urls.py`, checks that filters
and tags exist, and checks every variable against a declared map of what each
view actually puts in its context.

One specific trap it guards: `floatformat:"3u"` inside a CSS `style` attribute.
The `u` means unlocalized. Without it, a French locale renders `0,42` and the
`width: 0,42%` silently breaks the meter bars. Django's `add` filter also
concatenates when handed floats, which is why all geometry is computed in the
view in Python rather than in the template.

## What these do not cover

Worth saying out loud, because the honest answer to "is it tested?" is "these
four things are":

- **Detection accuracy.** That is what the trained model's validation metrics are
  for (mAP50, per-class), not for a unit test.
- **The severity thresholds.** They are a documented rule over measured mask
  area, not a learned model — the dataset carries no severity labels. The rule
  can be checked for self-consistency, which `test_multiview.py` partly does, but
  it cannot be validated against ground truth that does not exist.
- **Anything requiring a live Django process** — form submission, file upload,
  the PDF, the admin. Those are verified by running the server and using it.
