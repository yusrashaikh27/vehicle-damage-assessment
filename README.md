# Intelligent Vehicle External Damage Assessment and Cost Estimator

Upload photographs of a damaged car from one or more angles. The system segments
each damaged region with a YOLO instance-segmentation model, grades severity from
the area each mask covers, and returns a single itemised repair estimate with GST
at 18%.

Django + SQLite web app, a framework-free `core/` holding the assessment logic,
and a Django REST Framework API. Final-year BE major project, VTU.

---

## Start here: you need the dataset, and it is not in this repository

Four things are excluded by `.gitignore` because they total about 1.9 GB:

| Path | Size | How to get it |
| --- | --- | --- |
| `Detection.v1-cardd.yolov11/` | 388 MB | Re-download the export from Roboflow |
| `dataset/` | 171 MB | `python tools/prepare_dataset.py` |
| `dataset.zip` | 147 MB | `cd dataset && zip -r ../dataset.zip .` |
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

## Training

Training runs in Google Colab on a free T4 GPU, not locally — a MacBook has no
NVIDIA GPU and CPU training is 20–40x slower. `notebooks/train_colab.ipynb` is
ready to run; **`docs/TRAINING.md` walks through it step by step**, including what
every training flag means and what to do if the first run disappoints.

Short version: upload `dataset.zip` to `MyDrive/vda/`, open the notebook in Colab,
switch the runtime to a T4, run the cells. About 2h20m for 100 epochs (measured, 31 Aug 2026). Then
drop the resulting `best.pt` into `weights/` and restart the server.

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

## The train/test leak, and how it was fixed

An earlier version of this README said "do not report an accuracy figure until the
split is fixed", and estimated the leak at 12.1% of the test split. Both the fix
and the estimate have now been dealt with — the fix is in, and the estimate was far
too low.

The export's 10,000 files are **3,933 photographs**, each augmented 2–3 times by
Roboflow. Files sharing a filename stem before `_jpg.rf.` are the same photograph.
The export's own split cut across those families, so:

| | before | after |
| --- | --- | --- |
| test images that are a copy of a train photograph | **1,208 of 1,482 (81.5%)** | **0** |
| valid likewise | 1,198 of 1,483 (80.8%) | 0 |
| augmentation families spanning two splits | 48% | 0 |
| padded rotations in valid/test | ~72% | 0 |

81.5%, not 12.1%. An mAP measured on the old split would not have been inflated so
much as meaningless.

`tools/prepare_dataset.py` now splits by cluster rather than by image, keeps
augmented copies in train only, and refuses to declare success unless all three
leak assertions pass. Re-run it and it prints the checks; `dataset/` currently
holds a verified-clean 2,742 / 588 / 603 split with every class within a point of
70% train. **mAP measured on this split is reportable.**

One caveat to carry into the report: `tire_flat` has only 32 test instances, under
the 40 the script wants, so its per-class AP will swing between runs. Report it as
a range or with an explicit caveat. That is a limit of the data — 211 flat-tyre
instances exist in 3,933 photographs — not a bug in the split.

**The 12.1% figure came from a broken measurement, and so did a second claim that
was worse.** The old README warned: "do not merge every file that shares a filename
stem — 98.3% of same-stem pairs are genuinely different photographs". That is
false, and it pointed away from the actual fix. Both errors came from comparing
*raw* colour histograms and assuming rotation barely changes one. Rotation adds up
to 28% pure-black padding, which rewrites the histogram and pushes a
same-photograph pair below the acceptance threshold, so same-photograph pairs were
counted as different. Masking near-black pixels first separates the populations
completely: same-family pairs median 0.87 with none below 0.60, different-family
pairs median 0.37 with **none above 0.78**. `docs/DATASET_NOTES.md` has the
measurement, the control group, and the visual spot-check of the ambiguous cases.

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
tools/           prepare_dataset.py, setup_claude_cli.sh
docs/            RUNNING.md, TRAINING.md, DATASET_NOTES.md, PLAN.md
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
