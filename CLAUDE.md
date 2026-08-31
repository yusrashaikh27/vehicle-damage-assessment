# Intelligent Vehicle External Damage Assessment and Cost Estimator

VTU final-year BE major project. Django + SQLite web app; YOLO11-seg segments car damage,
severity comes from mask area, output is an itemised repair estimate with 18% GST.

**There is a written report and a viva.** Defensibility of claims matters as much as
working code. If evidence disagrees with a doc, show the measurement — do not just assert.

## Commands

    source venv/bin/activate            # Python 3.12 — NOT 3.14, no torch wheels
    python manage.py runserver          # then http://127.0.0.1:8000/ (use the IP)
    python tests/run_all.py             # 5 checks, ~1s, no GPU/db/network. Run before committing.
    python tools/prepare_dataset.py     # rebuilds dataset/ from the Roboflow export

Docs: `docs/RUNNING.md` (the app), `docs/TRAINING.md` (Colab), `docs/DATASET_NOTES.md`
(what the data actually is), `docs/PLAN.md` (schedule + pending report edits).

## Things that look like bugs but are deliberate — do not "fix" these

**The stub detector and its banner.** With no `weights/best.pt` the detector returns
invented regions and every page says so. That banner is honesty about fabricated numbers,
not a placeholder to remove.

**`core/` imports no Django, anywhere.** That is what lets severity and the whole cost
model be tested with no database, settings module or migrations, and why `run_all.py`
finishes in a second. Keep the boundary; `assessment/services.py` is the only bridge.

**Multi-view merge takes the maximum, not the sum** (`core.cost.estimate_combined`). A
bumper shot twice is one bumper to buy. The system cannot re-identify damage across views,
so max is the conservative reading — it can understate separate damage but cannot invent
damage.

**`area_fraction` has a per-frame denominator**, so detections are grouped by photograph
everywhere. Flattening them would place fractions with different denominators side by side.

**The cost baseline is a change detector, not a correctness test.** When a pricing change
is intended it goes red: read the diff, confirm every moved figure was meant to move, then
`python tests/cost_baseline.py write`. A permanently-red baseline detects nothing.

## The dataset: two traps that have already produced wrong numbers twice

`dataset/` is verified clean as of 2026-08-31 — train 2,742 / valid 588 / test 603, 3,933
photographs, 8,628 instances. **mAP measured on it is reportable.** Do not re-split it
without reading `docs/DATASET_NOTES.md` first.

**Trap 1 — filename families ARE augmentation families.** The export's 10,000 files are
3,933 photographs augmented 2–3x; files sharing the stem before `_jpg.rf.` are the same
photograph. Older versions of README.md and DATASET_NOTES.md claimed the opposite ("98.3%
of same-stem pairs are genuinely different photographs") and warned against grouping by
family. That was false and it hid an **81.5% train/test leak**. Cause: comparing *raw*
colour histograms, when rotation adds up to 28% black padding that rewrites them. Always
mask near-black pixels (luminance > 24) and always run a different-family control group —
the original analysis had no control, which is why it never noticed.

**Trap 2 — the split balances per-class instance counts, not image counts.** Valid/test
must contain only unpadded photographs, which pins 1,468 clusters to train *before*
dealing starts. Balance on image count and train looks over-full, so the rarest classes —
dealt first — get pushed into valid/test. That produced `tire_flat` 87 in train against 124
across valid+test, which is backwards. If you touch `stratified_group_split`, check the
`tire_flat` row is ~70/15/15, not just the image row.

`tools/prepare_dataset.py` asserts all three leak conditions and refuses to report success
otherwise. **Trust its assertions over any number remembered from a doc.**

## Accuracy claims

**Training has run and the test evaluation is done** — 100 epochs, 2.321 h on a Colab T4,
31 Aug 2026. Full figures and their caveats live in **`docs/RESULTS.md`**; read that before
quoting any number.

Headline, on the held-out test split (603 images, 1,299 instances): **mask mAP50 0.554 /
mAP50-95 0.401**. Quote **mask**, not box (0.575/0.427) — severity comes from polygon area,
so box mAP describes a number the pipeline never uses.

**The valid-split figures (0.567/0.407) are not reportable** — valid is what `best.pt`
selection watched. The 0.013 gap between them is a useful generalisation check but is **not**
leak evidence: leakage inflates test scores, so it would not show up as a gap. Cite
`prepare_dataset.py`'s assertions for the leak, and the per-class 70/15/15 table in
`docs/RESULTS.md` for split integrity.

**The load-bearing limitation: 84% of test instances belong to classes whose polygon area is
unreliable even when detection is correct.** Boundary quality (mAP50-95 ÷ mAP50) is 0.34 for
`crack`, 0.41 `dislocated_part`, 0.45 `scratch`, 0.50 `dent` — and those four are 84.1% of
instances. Since severity is area and cost is severity, that error lands straight in the
rupee figure. `crack` is worst per instance (mAP50 0.192); `scratch` is worst by exposure at
35.3% of instances. State this before an examiner finds it.

**Accuracy is *negatively* correlated with training volume — Spearman ρ = −0.64.** The three
best classes are the three rarest (`tire_flat` 147 train, `lamp_broken` 391, `glass_shatter`
411); `scratch` has the most data of any class (2,135) and is third worst. What predicts
accuracy is whether the damage has a boundary an annotator can agree on. **Never explain a
weak class with "not enough data"** — the data contradicts it.

`tire_flat` is written **≈0.96, not 0.963**: 32 test instances, recall 0.906 = 29/32, 95%
Wilson interval [0.76, 0.97]. Reproducible across two independent 32-instance samples, but
three decimals imply precision the sample cannot support.

The 92.5% / 98.5% / 94% figures in the report are Literature Survey numbers belonging to
*other papers*, on other datasets and mostly other metrics. Never present them as this
system's results and never compare 0.554 against them directly.

## Report edits still outstanding (see docs/PLAN.md)

1. Objective 2 says **YOLOv5s** → YOLO11-seg.
2. Methodology says **"bounding boxes"** → segmentation polygons. Load-bearing: severity is
   computed from polygon *area*, and a box has no usable area (a diagonal scratch fills
   ~15% of its box). As written, objective 3 does not follow from objective 2.
3. Abstract promises **two-wheelers** → CarDD is cars only. Drop it or state it as a
   limitation.
4. Two-wheelers must not be demoed.
