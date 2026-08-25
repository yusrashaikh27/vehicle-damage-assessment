# Dataset notes — what the CarDD export actually contains

Everything here was measured, not assumed. If a number in the report disagrees
with a number here, this file is the one that was checked. `tools/prepare_dataset.py`
turns the raw export into the clean `dataset/` used for training.

## Where it came from

`Detection.v1-cardd.yolov11/` is a Roboflow Universe export
(workspace `cardd-diezp`, project `detection-m16cd`, version 1, CC BY 4.0,
exported 16 October 2024). It is built on CarDD, a public car-damage dataset,
but it is not plain CarDD — see "It is a merge" below.

10,000 JPEG images, every one 640 px on its longest side, sRGB. Splits as
shipped: 7,000 train / 2,000 valid / 1,000 test.

## The labels are segmentation polygons, not boxes

Each label row is `class x1 y1 x2 y2 ... xn yn`, normalised to 0–1, with 4 to
96 vertices per polygon. The folder is named "Detection", which is misleading.

This is good news, and it decides a design question further down the pipeline:
a polygon has an **area**, so severity can be derived from how much of the panel
is affected rather than from a second model trained on severity labels that this
dataset does not contain. Train `yolo11*-seg`, not `yolo11*`.

## Instance counts (raw export)

| class | train | valid | test | total |
|---|---|---|---|---|
| crack | 1443 | 422 | 235 | 2100 |
| crash | 149 | 55 | 25 | 229 |
| dent | 3750 | 1057 | 540 | 5347 |
| dislocated part | 1072 | 338 | 161 | 1571 |
| glass shatter | 961 | 267 | 149 | 1377 |
| lamp broken | 970 | 252 | 133 | 1355 |
| no part | 606 | 178 | 86 | 870 |
| rub | 742 | 241 | 104 | 1087 |
| scratch | 5056 | 1518 | 654 | 7228 |
| tire flat | 346 | 102 | 39 | 487 |
| **total** | **15095** | **4430** | **2126** | **21651** |

Imbalance is real but manageable: 32:1 between the largest and smallest class.
Every class is trainable.

### A counting trap worth knowing about

9,652 of the 10,000 label files do not end in a newline. So this, the obvious
way to count, is **wrong**:

```bash
cat train/labels/*.txt | awk '{c[$1]++} END {for (k in c) print k, c[k]}'   # WRONG
```

`cat` glues each file's last row onto the next file's first row, losing about one
row per file — roughly 10,000 of 21,651 instances, and it distorts the per-class
distribution unevenly on top of that. Counting this way suggested `crash` had 4
instances when it has 229, which nearly led to a wrong decision about dropping
the class. Count per file instead:

```python
for p in Path("labels").iterdir():
    for line in p.read_text().splitlines():
        ...
```

## It is a merge of several annotation projects

The export has 10,000 files but only 3,969 distinct filename stems, which looks
like every photograph was duplicated 2–3 times. It is not.

Comparing image content instead of names: of a 250-pair sample of same-stem
files, 97% are **different photographs** (41% do not even share dimensions).
Filename stems also come in several conventions (`000123`, `8_jpeg`,
`akhand_b43_2`), which is the giveaway — this is several projects merged, each
numbering from 1.

Two consequences:

- Grouping by filename to "deduplicate" discards ~6,000 genuinely distinct
  images. Don't.
- Where two projects annotated the same photograph, they disagree. Among
  same-stem pairs: 54% draw the same damage with a different number of polygon
  vertices, 36% differ more substantially, and 8% assign **different classes**
  outright (one file says `lamp broken`, the other says `dislocated part` +
  `rub` + `rub`). This is the main argument for merging the taxonomy.

## Real duplicates, found from pixels

Detected with two signals that have to agree, because neither works alone:

- **256-bit dHash** (16×17 grayscale gradient hash) — sharp, but sensitive to
  cropping: a genuine re-crop of the same photo can score as high as 63.
- **512-bin RGB histogram intersection** — survives crops and rescaling, but too
  blunt alone, since many images are a grey panel on a black background.

A 64-bit dHash alone was tried first and was useless here: at hamming ≤ 6 it
flagged 413 images, and visual inspection showed only the hamming-0 pair was a
real duplicate. The rest were different cars sharing a silhouette. Close-up
damage crops on plain backgrounds have low entropy, so a coarse hash collides.

Accepting `dHash ≤ 64 AND histogram ≥ 0.88` finds **229 images in 112 duplicate
clusters** (10,000 images collapse to 9,883, so 117 copies are dropped), of which
**50 clusters straddled the export's own train/valid/test boundary** — genuine
leakage, about 1% of the data. Small, but free to fix.

## What `dataset/` contains

9,883 images (one per duplicate cluster), 21,415 instances, 7 classes, split
70/15/15 by cluster so no photograph can appear in two splits.

| class | train | valid | test | total |
|---|---|---|---|---|
| scratch | 5706 | 1229 | 1283 | 8218 |
| dent | 3786 | 764 | 745 | 5295 |
| dislocated_part | 1865 | 383 | 395 | 2643 |
| crack | 1449 | 296 | 333 | 2078 |
| glass_shatter | 942 | 201 | 212 | 1355 |
| lamp_broken | 933 | 205 | 205 | 1343 |
| tire_flat | 338 | 72 | 73 | 483 |
| **images** | **6918** | **1483** | **1482** | **9883** |
| of which no damage | 240 | 53 | 51 | 344 |

The 344 images with empty label files are background negatives. Keeping them
(3.5% of the set) helps suppress false positives on undamaged panels, which
matters for a system whose output is a repair bill.

### Why 10 classes became 7

Merged by the **repair action** each damage implies, because that is what the
cost estimator consumes:

| merged class | absorbs | repair action |
|---|---|---|
| scratch | scratch, rub | re-paint |
| dent | dent | pull out + re-paint |
| crack | crack | repair or replace part |
| lamp_broken | lamp broken | replace lamp unit |
| dislocated_part | dislocated part, no part, crash | refit or replace panel |
| glass_shatter | glass shatter | replace glass |
| tire_flat | tire flat | repair or replace tyre |

This removes exactly the distinctions the annotators disagreed about, and lifts
the smallest class from 229 instances to 483.

The counter-argument, worth being ready for: 10 classes are more granular and
all are trainable. `MERGE_GROUPS` in `tools/prepare_dataset.py` is a single list
— change it and the whole pipeline follows, since class names are read from the
generated `data.yaml`. Training both and comparing mAP would be a genuine
experimental result for the report if time allows.

## Two claims in the report this dataset cannot support

- **Two-wheelers.** The abstract promises "two-wheelers and four-wheelers".
  CarDD is cars only. A model trained on it will not detect motorcycle damage
  reliably. Either drop the claim or list it explicitly as a limitation.
- **Affected component.** The cost model in the report uses "the damaged
  component" (bumper, door, fender). No class here names a component — the
  labels are damage *types*. Component has to come from somewhere else: ask the
  user to pick the panel at upload, which also matches the report's existing
  "user uploads an image along with vehicle details".
