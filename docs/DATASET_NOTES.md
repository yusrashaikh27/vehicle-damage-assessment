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

## The 10,000 files are 3,933 photographs, augmented

The export has 10,000 files but only 3,969 distinct filename stems. Every name
follows Roboflow's augmentation convention:

```
<stem>_jpg.rf.<32 hex characters>.jpg
```

The stem is the source photograph; the hex is one augmented variant of it. So
files sharing a stem are the **same photograph**, augmented — an "augmentation
family". Grouping by family and keeping one member per family leaves **3,933
images**, which is the number that matters for every sample-size claim in the
report.

Independent corroboration that 3,933 is right: the CarDD paper (Wang et al.,
*IEEE T-ITS* 2023) documents roughly 4,000 images. Family grouping recovers the
original dataset's scale almost exactly. Any procedure that leaves ~9,900
"distinct" images has to explain where 6,000 extra photographs came from.

**7,211 of the 10,000 (72.1%) carry black padding** — rotation and shear leave
pure-black wedges in the corners, which is how an augmented copy is detectable
without comparing it to anything. Flip- and brightness-only augmentations leave no
padding, so 72.1% is a lower bound on the augmented share.

### An earlier version of this file said the opposite, and was wrong

It claimed 97% of same-stem pairs are *different photographs*, and told the reader
not to group by filename because doing so would discard ~6,000 real images. That
was measured, not invented — but the measurement was broken, and it is worth
recording how, because the same trap will catch anyone who re-checks this.

The check compared **raw** RGB histograms and assumed rotation barely changes a
histogram. It changes it enormously: a rotated copy gains up to 28% pure black,
which piles mass into the darkest bin and drags a same-photograph pair *below* the
0.88 acceptance threshold. Same-photograph pairs were therefore counted as
different photographs. The supporting detail that "41% do not even share
dimensions" has the same cause — rotating a 640×480 frame changes its dimensions.

Masking near-black pixels (luminance > 24) and renormalising separates the two
populations almost perfectly:

| pair type | median intersection | range |
|---|---|---|
| same family (n=250) | 0.87 | none below 0.60 |
| different families (n=250, control) | 0.37 | **none above 0.78** |

No overlap. The four hardest cases in the ambiguous 0.75–0.79 band were opened and
looked at: all four are one photograph rotated, and one pair even shares a
`12.04.2017` camera timestamp burnt into the frame.

The lesson is the control group. The original check measured same-stem pairs and
never asked what score *unrelated* pairs get, so there was nothing to compare
0.88 against.

## Clustering: filename families, then pixels

Two signals, unioned:

- **Signal A, filename family** — the stem before `_jpg.rf.`. Cheap and exact,
  and it accounts for 6,031 of the 6,067 merges.
- **Signal B, pixel similarity** — catches the same photograph submitted to two
  annotation projects under *different* names, which Signal A cannot see. Requires
  **256-bit dHash ≤ 64 AND black-masked histogram intersection ≥ 0.88**, both, since
  neither is reliable alone. Adds 36 more merges from 470 candidate pairs.

Signal B compares only the 3,969 family *representatives*, not all 10,000 images.
This is not just an optimisation. Rotated copies are largely black, so their
dHashes cluster tightly and comparing all pairs produces a candidate-pair
explosion — the first version of this ran for over six minutes and had not
finished. Comparing representatives brings it to about 20 seconds.

On earlier hash choices: a 64-bit dHash alone was useless here. At hamming ≤ 6 it
flagged 413 images and only the hamming-0 pair was a real duplicate; the rest were
different cars sharing a silhouette. Close-up damage crops on plain backgrounds
have low entropy, so a coarse hash collides.

Result: **3,933 clusters from 10,000 images.** 9,432 images sit in a cluster of two
or more, and **1,872 clusters straddled the export's own train/valid/test
boundary.**

## The leak in the export's own split, and the fix

Before this fix, `dataset/` inherited the export's split, and the numbers were bad:

| | before | after |
|---|---|---|
| test images that are a copy of a train photograph | **1,208 of 1,482 (81.5%)** | **0** |
| valid likewise | 1,198 of 1,483 (80.8%) | 0 |
| families spanning two splits | 48% | 0 |
| padded rotations in valid/test | ~72% | 0 |

An mAP measured on the "before" column is not inflated, it is meaningless — the
model would be scored almost entirely on photographs it trained on.

Two rules fix it, and `tools/prepare_dataset.py` asserts both before it will
declare success:

1. **Split by cluster, never by image.** Every copy of a photograph lands in one
   split.
2. **Valid and test contain only unpadded photographs.** Black corner wedges never
   occur when a user uploads a photo, so scoring on them measures the wrong
   distribution. 1,468 photographs whose every surviving copy is padded (Roboflow
   augmented only the export's train split, so a photograph living solely there has
   no clean copy) are pinned to train.

## What `dataset/` contains

3,933 images, one per cluster, 7 classes, split 70/15/15 by cluster.

| class | train | valid | test | total |
|---|---|---|---|---|
| scratch | 2135 | 461 | 458 | 3054 |
| dent | 1575 | 338 | 339 | 2252 |
| dislocated_part | 746 | 162 | 161 | 1069 |
| crack | 624 | 134 | 135 | 893 |
| glass_shatter | 411 | 89 | 89 | 589 |
| lamp_broken | 391 | 84 | 85 | 560 |
| tire_flat | 147 | 32 | 32 | 211 |
| **images** | **2742** | **588** | **603** | **3933** |
| of which no damage | 26 | 24 | 0 | 50 |
| of which still padded | 1468 | 0 | 0 | 1468 |

Every class is within a point of 70% train, which is deliberate: the split
balances each class's **instance count** separately, not the image count. An
earlier version balanced image counts, and because rule 2 pins 1,468 clusters to
train before dealing starts, train looked over-full and the rarest classes — dealt
first — were pushed into valid and test. `tire_flat` came out 87 train against 124
across valid and test, which is backwards; a model that barely sees flat tyres
scores near zero on them however many test instances exist.

**`tire_flat` has only 32 test instances**, below the 40 the script asks for, so it
prints a warning. Report its AP as a range or with an explicit caveat, not as a
point value. This is a limit of the data — there are only 211 flat-tyre instances
in 3,933 photographs — and the honest fix is more flat-tyre images, not a
different split.

The 50 images with empty label files are background negatives. Keeping them helps
suppress false positives on undamaged panels, which matters for a system whose
output is a repair bill.


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

The merge lifts the smallest class from `crash`'s 229 instances to `tire_flat`'s
211 after deduplication — which is not an improvement in itself, because `crash`'s
229 counted augmented copies. The real gain is that `crash`, `no part` and
`dislocated part` are the same repair job, so a model no longer has to learn a
three-way distinction that the cost estimator immediately throws away.

Note that an earlier version of this file justified the merge differently: it
claimed two annotation projects labelled the same photographs and disagreed about
classes 8% of the time. That rested on the same-stem misreading corrected above.
Same-stem files are one photograph augmented, and their labels differ because
rotation and shear clip instances at the frame edge — changing vertex counts, and
sometimes removing an instance entirely, which is what "different classes" was
really measuring. The repair-action argument is the honest one and stands on its
own; do not use the annotator-disagreement argument in the report or the viva.

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
