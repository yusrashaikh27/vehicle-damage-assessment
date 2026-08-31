# Results — measured, reportable figures

**Model:** YOLO11s-seg (10,069,525 parameters, 32.9 GFLOPs), fine-tuned from COCO weights.
**Training:** 100 epochs, 2.321 h, Google Colab Tesla T4, 31 Aug 2026.
**Evaluated:** 31 Aug 2026 on the **held-out test split** — 603 images, 1,299 instances.

Quote **mask** mAP, not box mAP. The system computes severity from polygon area, so mask
quality is what actually determines the output; box mAP describes a number the pipeline
never uses.

## Headline

| Metric | Box | **Mask** |
|---|---|---|
| mAP50 | 0.575 | **0.554** |
| mAP50-95 | 0.427 | **0.401** |

Precision 0.640, recall 0.540 (mask, all classes). Inference 9.7 ms/image on a T4
(≈103 images/s), so throughput is not a design constraint for the web app.

## Per class (mask)

Ordered worst to best. "Boundary" is mAP50-95 ÷ mAP50 — how well the outline survives a
stricter overlap requirement. "Share" is the class's fraction of all test instances.

| Class | mAP50 | mAP50-95 | Boundary | Share | Test inst. | Train inst. |
|---|---|---|---|---|---|---|
| crack | 0.192 | 0.066 | 0.34 | 10.4% | 135 | 624 |
| dislocated_part | 0.288 | 0.117 | 0.41 | 12.4% | 161 | 746 |
| scratch | 0.407 | 0.185 | 0.45 | 35.3% | 458 | 2,135 |
| dent | 0.501 | 0.253 | 0.50 | 26.1% | 339 | 1,575 |
| lamp_broken | 0.739 | 0.586 | 0.79 | 6.5% | 85 | 391 |
| glass_shatter | 0.788 | 0.677 | 0.86 | 6.9% | 89 | 411 |
| tire_flat | ≈0.96 | ≈0.92 | 0.96 | 2.5% | 32 | 147 |

`tire_flat` is deliberately written as ≈0.96 rather than 0.963. It rests on 32 instances
across 31 images; its recall of 0.906 means 29 of 32 found, and the 95% Wilson interval on
that is [0.76, 0.97]. The point estimate is reproducible — the validation split, an
independent 32 instances, gave 0.979/0.923 — but the interval is wide and quoting three
decimals implies precision the sample size cannot support.

## Validation vs test: the generalisation check

The validation split cannot be reported as a result, because it is what `best.pt` selection
and early stopping watched; it stopped being held out the moment it was used to choose a
checkpoint. It is still useful as a comparison.

| | Valid (588 img) | Test (603 img) | Δ |
|---|---|---|---|
| mask mAP50 | 0.567 | 0.554 | −0.013 |
| mask mAP50-95 | 0.407 | 0.401 | −0.006 |

Overall mask mAP50 falls by 0.013 — about 2% relative. Two things follow.

First, checkpoint selection did not meaningfully overfit the validation split. A model tuned
to a specific 588 images would drop much further on a fresh 603.

Second, the per-class movement is **not** in one direction: crack −0.079, lamp_broken
−0.057, glass_shatter −0.054, tire_flat −0.016, dislocated_part −0.007, but dent +0.059 and
scratch +0.061. A genuine generalisation gap pushes every class the same way. A mixture of
signs on samples this size is what sampling noise looks like, which is the more credible
reading of a 0.013 overall difference.

**What this does not prove.** A small valid-to-test gap is not itself evidence that the
train/test leak is fixed. Leakage inflates test scores rather than deflating them, so it
would not show up as a gap. The leak is established directly and separately, by the
assertions in `tools/prepare_dataset.py`, which refuse to write a dataset unless zero test
photographs share an augmentation family with any training photograph. Do not offer the gap
as leak evidence in the viva — offer the assertion, and offer the split table below.

## Split integrity

Counted from `dataset/labels/` on 2026-08-31, not from any document:

| Class | Train | Valid | Test | Train % | Valid % | Test % |
|---|---|---|---|---|---|---|
| scratch | 2,135 | 461 | 458 | 69.9 | 15.1 | 15.0 |
| dent | 1,575 | 338 | 339 | 69.9 | 15.0 | 15.1 |
| crack | 624 | 134 | 135 | 69.9 | 15.0 | 15.1 |
| dislocated_part | 746 | 162 | 161 | 69.8 | 15.2 | 15.1 |
| glass_shatter | 411 | 89 | 89 | 69.8 | 15.1 | 15.1 |
| lamp_broken | 391 | 84 | 85 | 69.8 | 15.0 | 15.2 |
| tire_flat | 147 | 32 | 32 | 69.7 | 15.2 | 15.2 |
| **total** | **6,029** | **1,300** | **1,299** | 69.9 | 15.1 | 15.1 |

Every class sits within 0.2 points of 70/15/15. This is the fix for the second dataset trap
(see `docs/DATASET_NOTES.md`): an earlier split balanced *image* counts and pushed the rarest
classes into valid/test, giving `tire_flat` 87 train against 124 in valid+test.

Ultralytics independently reported `Instances 1299` for the test split, matching the sum of
the per-class counts above exactly. Every polygon survived `prepare_dataset.py` → zip →
Drive upload → unzip → evaluation with none dropped or silently rejected.

## The load-bearing limitation

**For 84% of the damages this system sees, the polygon area it computes is unreliable even
when the detection itself is correct.**

Severity is derived from mask area and cost is derived from severity, so area error
propagates directly into the rupee figure. The boundary column above measures exactly this:
`crack` 0.34, `dislocated_part` 0.41, `scratch` 0.45 and `dent` 0.50 all lose more than half
their score when the overlap requirement tightens, and those four classes are 84.1% of test
instances. The model is finding these damages and drawing them wrongly.

Two classes deserve separate mention for different reasons. `crack` is the worst per
instance — lowest mAP50 at 0.192, worst boundary at 0.34, and the largest valid-to-test drop
of any class. `scratch` is the worst by exposure: its boundary quality is only marginally
better, but at 35.3% of instances it is the single largest source of cost error in the
system.

State this before an examiner finds it. The defensible position is that the system
demonstrates a working end-to-end pipeline whose detection stage is sound and whose
area-to-cost stage inherits a known, measured boundary error — not that it produces accurate
estimates.

## Accuracy does not track training volume

Spearman ρ between per-class training instance count and test mask mAP50 is **−0.64**. The
relationship is *negative*: the three most accurate classes are the three rarest
(`tire_flat` 147, `lamp_broken` 391, `glass_shatter` 411), while `scratch`, with 2,135
training instances — more than any other class by a factor of 1.4 — is third from bottom.

What predicts accuracy here is whether the damage has a definite boundary. A flat tyre, a
smashed lamp and shattered glass are discrete objects with an edge a human annotator can
agree on. A scratch and a crack are thin, low-contrast marks whose extent is a judgement
call, so the ground-truth polygons are themselves inconsistent and the model cannot do
better than the labels.

**Consequence for the report:** never explain a weak class with "insufficient training
data". The data contradicts it. More annotated scratches would not fix `scratch`; more
consistent scratch annotation might.

## Do not compare against the Literature Survey

The 92.5% / 98.5% / 94% figures in the report belong to other papers, on other datasets, and
mostly measure other things (classification accuracy, or box detection). Placing 0.554
beside them invites the conclusion that this system is worse, which the comparison cannot
support in either direction. Cite them as related work only.
