# Training the model

The app runs without a trained model — it falls back to a stub detector and shows a
banner saying so. Training is what replaces the stub with real detections, and it is
the only thing left on the critical path.

Training does **not** happen on your Mac. It happens in Google Colab on a borrowed
GPU, for one reason: a MacBook has no NVIDIA GPU, and YOLO training on a CPU is
roughly 20–40x slower. A run that takes 2h20m in Colab would take most of three
days locally.

## What you are about to do, in one paragraph

Upload `dataset.zip` to Google Drive, open `notebooks/train_colab.ipynb` in Colab,
switch the runtime to a T4 GPU, and run the cells top to bottom. About 2h20m
later you download one file, `best.pt`, drop it into `weights/`, restart the Django
server, and the placeholder banner disappears.

## Step 1 — upload the dataset (do this first, it takes the longest)

`dataset.zip` is in the project folder, 147 MB.

1. Go to [drive.google.com](https://drive.google.com).
2. Create a folder called **`vda`** in *My Drive* (exactly that name — the notebook
   looks for `MyDrive/vda/dataset.zip`).
3. Drag `dataset.zip` into it.

Leave it uploading and carry on reading.

If you ever regenerate the dataset, rebuild the zip before re-uploading:

    python tools/prepare_dataset.py
    cd dataset && zip -rq ../dataset.zip . && cd ..

## Step 2 — open the notebook in Colab

Go to [colab.research.google.com](https://colab.research.google.com) → **Upload** →
choose `notebooks/train_colab.ipynb`.

## Step 3 — get a GPU (skip this and nothing works)

**Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**

Colab gives you a CPU by default. Cell 1 exists purely to check this: it prints the
GPU name, or `NO GPU FOUND` if you forgot. Do not run cell 5 until cell 1 prints a
GPU.

## Step 4 — run cells 1 to 5

Shift+Enter runs a cell and moves to the next. In order:

| Cell | What it does | Watch for |
|---|---|---|
| 1 | Checks a GPU is attached | Must print `Tesla T4`, not `NO GPU FOUND` |
| 2 | `pip install ultralytics` | ~1 minute |
| 3 | Mounts Google Drive | A popup asks permission — allow it |
| 4 | Unzips the dataset to Colab's local disk | Should print 2742 / 588 / 603 images |
| 5 | **Trains.** ~2h20m (measured) | Leave the tab open |

Cell 4 unzips to Colab's local disk rather than reading from Drive, deliberately.
Training reads all 3,933 images once per epoch, 100 times over; reading them across
the network each time would take longer than the training itself.

Cell 4 also rewrites `path: .` in `data.yaml` to an absolute path. `data.yaml` ships
relative so it works on any machine, but ultralytics resolves relative dataset paths
against its own settings directory rather than the yaml's location, which produces a
confusing "dataset not found" pointing at a directory you have never heard of.

## What cell 5's settings mean

```python
model = YOLO("yolo11s-seg.pt")
```

**`yolo11s-seg.pt`** — YOLO11, size **s** (small), **seg** for segmentation. We start
from weights already trained on COCO, a large general-purpose photo dataset, and
fine-tune them on car damage. Starting from COCO rather than from scratch is worth
roughly ten times the data: the early layers already know edges, texture and
curvature, and none of that needs relearning.

**`-seg` is not optional.** Segmentation predicts a polygon outline; plain detection
predicts a rectangle. Severity in this project is computed from the *area* a damage
covers, and a rectangle's area is not the damage's area — a diagonal scratch fills
maybe 15% of its bounding box. Objective 3 depends on objective 2 producing polygons.

```python
epochs=100      imgsz=640     batch=16
workers=2       patience=15   seed=0
```

**`epochs=100`** — one epoch is one pass over all 2,742 training images. The model
sees everything 100 times, improving a little each pass. This is the main time dial:
100 epochs ≈ 2h20m, so halve it if you are short on time.

**`imgsz=640`** — every image is resized to 640×640. The dataset is already 640 on
its long edge, so nothing is upscaled. Larger would be sharper on small damage and
much slower; smaller would be faster and would start losing fine scratches.

**`batch=16`** — 16 images go through the GPU at once. This is a memory setting, not
a quality one. 16 fits comfortably in a T4's 15 GB. If you ever see
`CUDA out of memory`, drop it to 8.

**`workers=2`** — background processes loading images. Colab free gives 2 CPU cores,
so more workers would fight each other for them.

**`patience=15`** — stop early if validation accuracy has not improved for 15
consecutive epochs. This is why a generous epoch budget is safe: if the model stops
learning at epoch 62, you get a finished run at epoch 77 rather than sitting through
38 pointless ones.

**`seed=0`** — fixes the randomness so the run is reproducible. If someone in the
viva asks whether your numbers can be regenerated, they can.

**`plots=True`** — writes the confusion matrix and precision-recall curves. These are
report figures; you want them.

## What you will see while it trains

One line per epoch, roughly:

```
    Epoch  GPU_mem  box_loss  seg_loss  cls_loss  Instances     Size
    12/100   6.2G     1.203     1.891     0.842        847      640
```

The three losses measure how wrong the model currently is — box position, mask shape,
and class. **They should trend downward.** They will bounce around between epochs;
that is normal. What matters is the direction over ten epochs, not one.

Every ten epochs or so it validates and prints mAP. **mAP** (mean Average Precision)
is the standard object-detection score, 0 to 1, higher is better. `mAP50` is the
lenient version (a prediction counts if it overlaps the truth by 50%); `mAP50-95` is
the strict average across overlap thresholds from 50% to 95%. `mAP50-95` is always
much lower and is the more honest number.

A `[mirrored checkpoint to Drive]` line appears after each save. That is your
insurance — see below.

## If Colab disconnects

It will happen; the free tier is not guaranteed. It costs you **one epoch**, because
every checkpoint is copied to Drive as it is written.

Reconnect, re-run cells 1–4, then run **cell 7** (the resume cell) instead of cell 5.
It pulls `last.pt` back from Drive and continues with the optimiser state and
learning-rate schedule intact.

To reduce the odds: keep the tab visible, do not let the laptop sleep, and do not run
two Colab notebooks at once.

## Step 5 — evaluate (cell 8)

This is the cell that produces the numbers for your report. It runs the trained model
against the **test** split — photographs it has never seen, and now provably so.

It prints overall `mAP50` and `mAP50-95` for boxes and masks, then a per-class
breakdown. Two things to note when you write them up:

- Quote **mask** mAP as the headline, not box mAP. The masks are what the system
  actually uses.
- **`tire_flat` has only 32 test instances.** Its AP will swing by several points
  between runs. Report it as a range or with an explicit caveat. This is a property
  of the data — 211 flat-tyre instances exist across 3,933 photographs — not a bug.

Do not write any accuracy figure into the report before this cell has run. The three
percentages currently in the report (92.5%, 98.5%, 94%) belong to *other papers* in
your Literature Survey, and must not be read as this system's results.

## Step 6 — bring the weights home (cells 9 and 10)

Cell 9 copies everything to `MyDrive/vda/runs/` — weights, `results.csv`, and the
plots. Cell 10 is optional and renders a few annotated test predictions, which make
good report figures.

Then, on your Mac:

1. Download `best.pt` from `MyDrive/vda/`.
2. Put it at `weights/best.pt` in the project folder (create `weights/` if needed).
3. Restart the server: `python manage.py runserver`.

The placeholder banner disappears on its own, and existing inspections can be
re-assessed from the admin's re-run action.

Also worth downloading for the report: `confusion_matrix.png`, `PR_curve.png` and
`results.png` from `runs/yolo11s-seg/`.

## If the first run is disappointing

Look at the confusion matrix before changing anything — it tells you *which* classes
are failing, and the fix differs.

- **One rare class is bad, everything else is fine** → a data problem, not a training
  problem. More epochs will not invent flat-tyre photographs.
- **Everything is mediocre** → try `epochs=150`, or step up to `yolo11m-seg.pt`
  (medium). Roughly 2.5x slower per epoch, usually a few mAP points better.
- **Two classes are confused with each other** → check whether they are genuinely
  distinguishable in the images. `crack` and `scratch` on a bumper are hard for
  people too, and that is a legitimate finding to report rather than a failure.
- **Losses never came down at all** → something is wrong with the data path. Re-check
  cell 4's output printed 2742 / 588 / 603.

Keep the first run's `results.csv` whatever happens. A comparison between two runs is
a stronger report section than a single number.
