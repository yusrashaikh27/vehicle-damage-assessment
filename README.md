# Intelligent Vehicle External Damage Assessment and Cost Estimator

Upload photographs of a damaged car from one or more angles. The system segments
each damaged region with a YOLO instance-segmentation model, grades severity from
the area each mask covers, and returns a single itemised repair estimate with GST
at 18%.

Django + SQLite web app, a framework-free `core/` holding the assessment logic,
and a Django REST Framework API. Final-year BE major project, VTU.

---

## Start here: you need the dataset, and it is not in this repository

Four things are excluded by `.gitignore` because they total about 2.1 GB:

| Path | Size | How to get it |
| --- | --- | --- |
| `Detection.v1-cardd.yolov11/` | 388 MB | Re-download the export from Roboflow |
| `dataset/` | 384 MB | `python tools/prepare_dataset.py` |
| `dataset.zip` | 345 MB | `cd dataset && zip -r ../dataset.zip .` |
| `venv/` | 1.2 GB | `python3.12 -m venv venv` (see below) |

**The app runs without any of them.** You only need the dataset to retrain the
model. With no trained weights at `weights/best.pt` the detector falls back to a
stub and every page shows a banner saying the detections are placeholders — so
you can set up, click through the whole flow, and see a real itemised estimate
without downloading a single gigabyte.

To get the dataset: ask Yusra for the Roboflow export link, unzip it into the
project root as `Detection.v1-cardd.yolov11/`, then run
`python tools/prepare_dataset.py`. That script does the merging, deduplication and
splitting and writes `dataset/`. Read `docs/DATASET_NOTES.md` first — it documents
three counting traps in the raw export that produce silently wrong numbers if you
trust the obvious approach.

## Setup

Python **3.12** specifically. Not 3.14 — torch and ultralytics do not publish
reliable wheels for it, and building them from source is not a thing to discover
the week before a submission.

```
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Then open **http://127.0.0.1:8000/** — use the IP, not `localhost`. On some macOS
setups `localhost` resolves to IPv6 `::1` while the dev server binds IPv4, which
looks like the server failed to start when it is running perfectly.

`migrate` is enough on a fresh clone; `0001_initial.py` is committed, so nobody
needs to re-generate the schema.

## The four pages

| URL | What it is |
| --- | --- |
| `/` | Front page: owner details, vehicle details, one upload slot per walk-around angle |
| `/inspections/` | Every inspection so far, searchable by plate, owner, make or model |
| `/admin/` | Django admin — the report's Admin Panel layer |
| `/api/inspections/` | REST API, browsable in the browser |

A result page is at `/inspections/<id>/`, its PDF at
`/inspections/<id>/report.pdf`. Both are linked from the pages above.

## Before you demo or commit

```
python tests/run_all.py
```

Five checks, about a second, no GPU, database or network needed. A 155-case cost
regression baseline, four multi-view property tests, and two AST-based checkers
that resolve every model attribute the admin, API and templates ask for. See
`tests/README.md` for what each one proves and what they deliberately do not
cover.

The cost baseline is a **change detector, not a correctness test**. When a pricing
change is intended it will go red; read the diff, confirm every figure that moved
was meant to move, then re-record with `python tests/cost_baseline.py write`. A
permanently-red baseline detects nothing.

## Trying it out

`sample_uploads/` holds eight photographs, one per damage class, with a README
giving vehicle details to type and three runs with pre-computed totals. The
images themselves are not committed (they are derived from the excluded dataset)
but `sample_uploads/README.md` is, and it tells you how to pick your own.

The run worth doing first: one photograph gives ₹4,278, and *three* photographs
all labelled the same panel give **the same ₹4,278**. That is the central design
decision — see below.

## Two design decisions most likely to be questioned

**Views are merged before pricing, taking the maximum and not the sum**
(`core.cost.estimate_combined`). A bumper photographed from the front and again
from the front-left is one bumper to buy, paint once, and fit once. The system
does not re-identify damage across views, so it cannot distinguish one dent
photographed twice from two separate dents — the maximum is the conservative
reading, because it can understate genuinely separate damage but cannot invent
damage that is not there. Labour is sublinear rather than flat, since a second
damaged area on the same panel does add work.

**`area_fraction` has a per-frame denominator**, so detections are grouped by
photograph everywhere — result page, PDF, API and admin — rather than flattened
into one list. A flat list would place fractions with different denominators side
by side and invite a reader to compare or sum them, which would be meaningless.

**The component is supplied by the user, not predicted.** No class in the dataset
names a component; the labels are damage *types* (scratch, dent, crack…). Since
the part price depends entirely on which panel is damaged, the panel has to come
from somewhere, and asking is both more honest and more accurate than guessing.
Each angle pre-selects the component it usually shows, and every one is editable.

## Known issue: do not report an accuracy figure until the split is fixed

Measured 2026-08-25, before any training run:

- **79.4% of `dataset/` is Roboflow augmentation copies**, not distinct
  photographs. Rotation and shear pad the frame with black, which is how they are
  detectable. Flip- and brightness-only augmentations leave no padding, so 79.4%
  is a lower bound.
- **12.1% of the test split (179 of 1,482 images) is a rotated copy of a
  photograph in train or valid**, and 11.9% of the validation split likewise.
  That is a real train/test leak: the deduplication keyed on a perceptual hash
  distance of ≤64, and a rotation moves that distance to roughly 100 while barely
  changing the colour histogram, so rotated copies passed straight through.

Any mAP measured on the current split is inflated and must not go in the report.
The fix belongs in `tools/prepare_dataset.py`, in two parts: exclude augmented
copies from validation and test, since padded rotations never occur at inference
anyway (this alone resolves 171 of the 179), and split by augmentation family
rather than by image, grouping on the filename stem before `.rf.` (needed for the
remaining 8, which are original photographs whose rotations landed elsewhere).

Do **not** try to fix this by merging every file that shares a filename stem.
98.3% of same-stem pairs are genuinely different photographs — the raw export
merged several annotation projects with colliding names, and collapsing them
would discard roughly 6,000 real images. `docs/DATASET_NOTES.md` has the details.

## Layout

```
core/            assessment logic - no Django imports anywhere
  damage_config.py   panels, angles, rate card, repair actions
  severity.py        area thresholds -> minor / moderate / severe
  cost.py            pricing, and the multi-view merge
  detector.py        YOLO wrapper, with the stub fallback
  pipeline.py        orchestration
  report.py          PDF generation
assessment/      the Django app; services.py is the only bridge to core/
vdac/            project settings and root URLs
tests/           five checks, no Django or GPU needed
tools/           prepare_dataset.py
docs/            RUNNING.md, DATASET_NOTES.md, PLAN.md
notebooks/       train_colab.ipynb - training runs in Colab, not locally
```

`core/` importing no Django is deliberate: it means severity grading and the
whole cost model can be tested with no database, no settings module and no
migrations, which is why `tests/run_all.py` finishes in about a second.

## Configuration

All optional; the defaults are correct for local development.

| Variable | Default | Notes |
| --- | --- | --- |
| `DJANGO_SECRET_KEY` | insecure dev value | Must be set before deploying anywhere real |
| `DJANGO_DEBUG` | `True` | Set `False` in production |
| `DJANGO_ALLOWED_HOSTS` | `127.0.0.1,localhost` | Comma-separated |
| `YOLO_WEIGHTS_PATH` | `weights/best.pt` | Point at weights kept outside the repo |
