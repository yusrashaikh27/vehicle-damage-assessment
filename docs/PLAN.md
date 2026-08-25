# Build plan — Intelligent Vehicle External Damage Assessment and Cost Estimator

Target: working, demonstrable system in 2–3 weeks from 22 Aug 2026.
Everything below traces back to the four objectives in `final report2.pdf`.

## The one scheduling decision that matters

Training is the long pole: ~3–4 hours per run on a free Colab T4, and you will
want two or three runs. Nothing else in the project depends on the trained
weights being *good* — only on them existing. So training starts first and runs
in the background while the Django app is built against a placeholder model. Do
not build the app first and train at the end; that is how projects run out of
time with an untrained model and no fallback.

## How the report's architecture maps onto what we build

The report describes six layers. Most of them already have an obvious Django
home, which is why Django is the right choice here — that, and you already know
it from the RAG chatbot.

| Report layer | What we actually build |
|---|---|
| User / Client Layer | Django template: upload form (image + vehicle make/model/year/panel) and a results page |
| Backend API Layer | Django views + DRF endpoints; file-type and size validation |
| Core Processing Layer | `core/` module: preprocess → detect → severity → cost → PDF report |
| Data Storage Layer | SQLite via the Django ORM; uploaded and annotated images on disk under `media/` |
| External Services | left out — see "Scope" below |
| Admin Panel | Django admin. Registering the models covers all five admin bullets in the report almost for free |

## Scope: what we are deliberately not building

The report lists MySQL, Redis caching, a payment gateway, Google Maps for nearby
service centres, and a mobile app. In the report's own wording these are hedged
("Redis **can be** used", "the system **can integrate** external services"), so
omitting them is consistent with what was written. If asked in the viva, the
honest answer is that they are integration surface, not research contribution,
and the time went into the parts that determine whether the system works.

SQLite instead of MySQL for the same reason: identical ORM code, one fewer
service to install on a demo machine. Swapping it is a settings change.

Two claims in the report that the *data* cannot support, and what to do:

- **Two-wheelers.** CarDD is cars only. Either drop the claim from the abstract
  or list it as a stated limitation. Do not demo a motorcycle.
- **Affected component.** The cost model needs to know it is a bumper rather than
  a door, but no class in the dataset names a component — the labels are damage
  types. The user selects the panel at upload, which the report already implies
  ("uploads an image along with vehicle details").

## Week 1 (22–28 Aug) — data, environment, training, skeleton

- [x] **Dataset repaired.** `tools/prepare_dataset.py` → `dataset/`: 9,883
      images, 21,415 instances, 7 classes, splits provably free of duplicate
      photographs. Findings in `docs/DATASET_NOTES.md`.
- [ ] **Rebuild the venv on Python 3.12.** The existing one is 3.14.6; torch and
      ultralytics wheels don't reliably support it yet.
- [ ] **Zip `dataset/` to Drive and start training** `yolo11s-seg`, 640 px,
      ~80 epochs, checkpointing to Drive so a Colab disconnect can resume.
- [ ] **Django skeleton**: project, `Inspection` and `Detection` models, upload
      view, media handling, admin registration. Runs end to end against a
      stock pretrained model so the plumbing is proven before real weights land.

## Week 2 (29 Aug – 4 Sep) — the three modules that are actually yours

- [ ] **Severity from mask area.** The dataset has no severity labels, so
      severity cannot be learned from it. It is computed: the polygon area of
      each detection as a fraction of the image, banded into Minor / Moderate /
      Severe with per-class thresholds (a shattered windscreen is never
      "minor"; a flat tyre is effectively binary). Thresholds live in one
      config dict, documented, so the rule is inspectable rather than magic.
      **Be upfront that this is a documented heuristic, not a trained
      classifier** — claiming otherwise is the kind of thing a viva finds.
- [ ] **Cost estimation.** `cost = part_cost(panel, vehicle_segment) ×
      replace_fraction(damage_class, severity) + labour_rate × hours(damage_class,
      severity)`, driven by a rate table you can cite a source for. This is
      exactly why the classes were merged by repair action.
- [ ] **Report generation.** PDF with the annotated image, per-damage table
      (type, severity, panel, cost line), and total. Objective 4.
- [ ] **Retrain** with whatever the first run's confusion matrix suggests.

## Week 3 (5–11 Sep) — evidence and defence

- [ ] **Evaluation on the held-out test split**: mAP50, mAP50-95, per-class AP,
      confusion matrix, precision/recall. These are the numbers that go in the
      report, and they are honest because the split is clean.
- [ ] **Limitations section**: cars only, severity is a heuristic, cost table is
      indicative, single-view images.
- [ ] **Viva preparation**: for each design decision, the alternative that was
      rejected and why. The dataset investigation is the strongest material here
      — you found and fixed leakage in a public dataset.
- [ ] Optional: 10-class vs 7-class mAP comparison; deployment.

## Objectives traceability

| Report objective | Delivered by |
|---|---|
| 1. Automated detection from uploaded images | Upload view + detection module |
| 2. Preprocess and identify dents, scratches, cracks, broken parts | `dataset/` (7 repair-action classes) + trained `yolo11s-seg` |
| 3. Classify severity and estimate repair cost | Severity module (mask area) + cost module (rate table) |
| 4. Generate and store a user-friendly report | PDF generator + `Inspection` model + Django admin |

Objective 2 names **YOLOv5s**. We use **YOLO11-seg** instead, for a reason worth
stating in the viva: the annotations are polygons, and polygons give an area,
which is what makes objective 3's severity assessment possible without severity
labels. Update that line in the report to say YOLO11.
