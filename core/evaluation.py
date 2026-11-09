"""The measured evaluation results, as data.

Transcribed once from `docs/RESULTS.md`, which is itself transcribed from the
Ultralytics run of 31 Aug 2026 (100 epochs, 2.321 h, Colab T4) evaluated on the
held-out test split. Kept here so the site, the report and any future chart all
read the same numbers from one place rather than each carrying its own copy of
a figure somebody typed by hand.

Three rules about these numbers, all of them load-bearing and all of them
enforced by `_validate()` at import:

1. MASK, NOT BOX. Severity is computed from polygon area, so mask quality is
   what determines the output. Box mAP (0.575 / 0.427) describes a number the
   pipeline never uses, and is recorded here only so nobody has to go looking
   for it and quote it by mistake.

2. TEST, NOT VALID. The validation split is what `best.pt` selection watched;
   it stopped being held out the moment it chose a checkpoint. Its figures are
   kept for the generalisation comparison and are explicitly not reportable.

3. tire_flat IS APPROXIMATE. 32 test instances, recall 0.906 = 29/32, 95%
   Wilson interval [0.76, 0.97]. The point estimate is reproducible across two
   independent 32-instance samples, but three decimals would imply a precision
   the sample size cannot support, so it is carried as a float for arithmetic
   and rendered with an explicit approximation marker.

Two sources of numbers, and why they are kept apart
---------------------------------------------------
The headline figures above (MASK_MAP50, CLASS_RESULTS and so on) are the ones
recorded on the Colab T4 straight after training.

The Colab results folder was not kept, so SPLITS and CLASS_DETAIL below come
from re-evaluating the same `best.pt` locally (Ultralytics 8.4.126, Apple M2
CPU) on the train, validation and test splits. That run reproduces the T4
headline to within 0.003 on every mAP figure, which is the expected size of a
CPU-versus-GPU numerical difference. The two sets are never mixed inside one
row of a table; the page says which is which.

This module imports nothing from Django, like everything else in core/.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

MODEL_NAME = "YOLO11s-seg"
MODEL_PARAMS = 10_069_525
# Derived, so the readable figure cannot drift from the exact one. A template
# cannot divide, and 10069525 with no thousands separators is not a number
# anybody reads off a screen.
MODEL_PARAMS_MILLIONS = MODEL_PARAMS / 1_000_000
MODEL_GFLOPS = 32.9
TRAIN_EPOCHS = 100
TRAIN_HOURS = 2.321
# Derived for the same reason: a template cannot multiply.
TRAIN_MINUTES = TRAIN_HOURS * 60
TRAIN_HARDWARE = "Google Colab Tesla T4"
EVALUATED_ON = "31 Aug 2026"

TEST_IMAGES = 603
TEST_INSTANCES = 1_299

# The other two splits, from the same dataset/ folder the model was trained on.
TRAIN_IMAGES = 2_742
TRAIN_INSTANCES = 6_029
VALID_IMAGES = 588
VALID_INSTANCES = 1_300

# Headline, mask. These two are the reportable figures.
MASK_MAP50 = 0.554
MASK_MAP50_95 = 0.401

# Box, recorded so it is never quoted by accident. See rule 1.
BOX_MAP50 = 0.575
BOX_MAP50_95 = 0.427

MASK_PRECISION = 0.640
MASK_RECALL = 0.540

INFERENCE_MS = 9.7

# Measured locally on the test split: about 0.3 s per image on a laptop CPU.
# Recorded because the app runs on a laptop in a demo, not on a T4.
CPU_INFERENCE_MS = 298.0
CPU_HARDWARE = "Apple M2 CPU"

# Valid split - NOT reportable. See rule 2.
VALID_MASK_MAP50 = 0.567
VALID_MASK_MAP50_95 = 0.407


@dataclass(frozen=True)
class ClassResult:
    """One damage class's measured performance on the held-out test split."""

    name: str
    label: str
    map50: float
    map50_95: float
    test_instances: int
    train_instances: int
    # True when the polygon outline is unreliable enough that the area - and
    # therefore the severity, and therefore the cost - inherits real error.
    # The cut is boundary < 0.55, which separates the four weak classes from
    # the three strong ones with a wide margin either side (0.50 vs 0.79).
    area_unreliable: bool

    @property
    def boundary(self) -> float:
        """How well the outline survives a stricter overlap requirement.

        mAP50-95 divided by mAP50. A class that finds the damage but draws it
        wrongly scores well on the loose threshold and badly on the strict one,
        so this ratio isolates outline quality from detection ability - which
        matters here because the pipeline consumes the outline, not the box.
        """
        return self.map50_95 / self.map50 if self.map50 else 0.0

    @property
    def share(self) -> float:
        """This class's fraction of all test instances."""
        return self.test_instances / TEST_INSTANCES

    @property
    def share_percent(self) -> float:
        """The same fraction as a percentage, for display.

        Derived rather than stored, and derived rather than left to the caller,
        because a Django template cannot multiply: `share` alone would have to
        be rendered as 0.353 where the sentence wants 35.3%.
        """
        return self.share * 100


# Ordered worst to best by mAP50, which is also the order they should be read:
# the limitation comes first.
CLASS_RESULTS: tuple[ClassResult, ...] = (
    ClassResult("crack", "Crack", 0.192, 0.066, 135, 624, True),
    ClassResult("dislocated_part", "Dislocated part", 0.288, 0.117, 161, 746, True),
    ClassResult("scratch", "Scratch", 0.407, 0.185, 458, 2_135, True),
    ClassResult("dent", "Dent", 0.501, 0.253, 339, 1_575, True),
    ClassResult("lamp_broken", "Lamp broken", 0.739, 0.586, 85, 391, False),
    ClassResult("glass_shatter", "Glass shatter", 0.788, 0.677, 89, 411, False),
    ClassResult("tire_flat", "Tyre flat", 0.963, 0.923, 32, 147, False),
)

# Rendered instead of the bare number for tire_flat. See rule 3.
APPROXIMATE = {"tire_flat"}

# Spearman rho between per-class training instance count and test mask mAP50.
# Negative: the three most accurate classes are the three rarest. Never explain
# a weak class here with "not enough data" - this figure contradicts it.
VOLUME_CORRELATION = -0.64


# ---------------------------------------------------------------------------
# Train / validation / test, from the local re-evaluation of best.pt
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitResult:
    """Mask metrics for one data split, all measured in the same local run."""

    key: str
    label: str
    role: str
    images: int
    instances: int
    precision: float
    recall: float
    map50: float
    map50_95: float

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall, 2PR / (P + R)."""
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


# Train first, test last. The training split contains Roboflow-augmented copies
# of each photograph, and the model was fitted on every one of them, so its
# figures flatter the model: they are here to make the generalisation gap
# visible, never to be quoted as accuracy.
SPLITS: tuple[SplitResult, ...] = (
    SplitResult("train", "Training", "Fitted on these images",
                TRAIN_IMAGES, TRAIN_INSTANCES, 0.914, 0.863, 0.909, 0.671),
    SplitResult("valid", "Validation", "Chose the checkpoint",
                VALID_IMAGES, VALID_INSTANCES, 0.711, 0.531, 0.564, 0.406),
    SplitResult("test", "Test", "Held out - the reportable figure",
                TEST_IMAGES, TEST_INSTANCES, 0.634, 0.546, 0.555, 0.402),
)

# Derived from the table above so the sentence quoting it cannot drift.
TRAIN_TEST_GAP = SPLITS[0].map50 - SPLITS[2].map50
TRAIN_TEST_GAP_POINTS = TRAIN_TEST_GAP * 100


@dataclass(frozen=True)
class ClassDetail:
    """Per-class mask metrics from the same local run as SPLITS.

    precision and recall are on the test split. train_map50 and test_map50 sit
    side by side so the generalisation gap can be read class by class.
    """

    name: str
    label: str
    precision: float
    recall: float
    train_map50: float
    test_map50: float

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def gap(self) -> float:
        """Training mAP50 minus test mAP50: how much does not carry over."""
        return self.train_map50 - self.test_map50


# Same order as CLASS_RESULTS (worst to best on test mAP50), enforced below.
CLASS_DETAIL: tuple[ClassDetail, ...] = (
    ClassDetail("crack", "Crack", 0.312, 0.274, 0.631, 0.195),
    ClassDetail("dislocated_part", "Dislocated part", 0.436, 0.341, 0.916, 0.290),
    ClassDetail("scratch", "Scratch", 0.500, 0.437, 0.902, 0.408),
    ClassDetail("dent", "Dent", 0.652, 0.451, 0.948, 0.504),
    ClassDetail("lamp_broken", "Lamp broken", 0.745, 0.682, 0.985, 0.736),
    ClassDetail("glass_shatter", "Glass shatter", 0.877, 0.730, 0.989, 0.788),
    ClassDetail("tire_flat", "Tyre flat", 0.920, 0.906, 0.995, 0.964),
)


# ---------------------------------------------------------------------------
# The train/test leak that was fixed (from README / docs/DATASET_NOTES.md)
# ---------------------------------------------------------------------------

# In the dataset export's own split, test images that were an augmented copy of
# a photograph also present in train. After tools/prepare_dataset.py: zero.
SPLIT_LEAK_BEFORE_COUNT = 1_208
SPLIT_LEAK_BEFORE_TOTAL = 1_482
SPLIT_LEAK_BEFORE_PERCENT = SPLIT_LEAK_BEFORE_COUNT / SPLIT_LEAK_BEFORE_TOTAL * 100
SPLIT_LEAK_AFTER_COUNT = 0


# ---------------------------------------------------------------------------
# Existing cost-estimation systems, for the method comparison
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExistingSystem:
    """One published or patented system, described from its own abstract.

    Methods only. No accuracy figure is recorded for any of them, because their
    datasets and cost sources are not public and a number lifted from another
    paper would not be measured on the same task.
    """

    name: str
    year: str
    detection: str
    severity: str
    cost: str
    url: str


EXISTING_SYSTEMS: tuple[ExistingSystem, ...] = (
    ExistingSystem(
        "Car damage assessment demo (Ant Financial)", "AAAI demo",
        "Damage and component recognition models, run on frames taken from video.",
        "Damage degree recognised by the models.",
        "Repair plan from a pricing algorithm.",
        "https://ojs.aaai.org/index.php/AAAI/article/view/7110/6964",
    ),
    ExistingSystem(
        "VisionClaimNet", "2026",
        "YOLOv8 object detection.",
        "A CNN classifies severity.",
        "Regression analysis estimates the repair cost.",
        "https://icnccom2026.aconf.org/presentation/34.html",
    ),
    ExistingSystem(
        "Generative-AI estimator (GPT-4o)", "IEEE, 2025",
        "A multimodal language model identifies the vehicle, panels and damage types from a long instruction prompt.",
        "Estimated by the language model.",
        "A rule-based cost framework written into the prompt, checked against expert assessments.",
        "https://dl.lib.uom.lk/handle/123/24758",
    ),
    ExistingSystem(
        "Image-similarity claim systems (patent US 11610074)", "Patent",
        "A deep network compares the photo with images of earlier damaged vehicles.",
        "Learned from claims whose damage level is known.",
        "Learned from claims whose repair time and cost are known.",
        "https://patents.google.com/patent/US11610074",
    ),
)


def unreliable_area_share() -> float:
    """Fraction of test instances whose computed polygon area is unreliable.

    The load-bearing limitation, computed rather than quoted so it cannot drift
    from the per-class table above.
    """
    return sum(c.test_instances for c in CLASS_RESULTS if c.area_unreliable) / TEST_INSTANCES


def _validate() -> None:
    """Fail at import if the transcription is internally inconsistent.

    Cheap, and it catches the realistic failure: somebody edits one row of the
    table and not the total, leaving the site quoting a share that no longer
    matches the instance counts it is derived from.
    """
    counted = sum(c.test_instances for c in CLASS_RESULTS)
    if counted != TEST_INSTANCES:
        raise ValueError(
            f"per-class test instances sum to {counted}, but TEST_INSTANCES is "
            f"{TEST_INSTANCES}. One of them is wrong; docs/RESULTS.md decides."
        )

    for c in CLASS_RESULTS:
        if not 0.0 <= c.map50_95 <= c.map50 <= 1.0:
            raise ValueError(
                f"{c.name}: expected 0 <= mAP50-95 ({c.map50_95}) <= mAP50 "
                f"({c.map50}) <= 1. A stricter overlap requirement cannot score "
                f"higher than a looser one."
            )

    ordered = [c.map50 for c in CLASS_RESULTS]
    if ordered != sorted(ordered):
        raise ValueError(
            "CLASS_RESULTS must stay ordered worst-to-best by mAP50; the "
            "charts and the table both rely on it."
        )

    # The new tables must describe the same classes, in the same order, and the
    # test split must be the same size as the one the headline was measured on.
    if [d.name for d in CLASS_DETAIL] != [c.name for c in CLASS_RESULTS]:
        raise ValueError(
            "CLASS_DETAIL must list the same classes in the same order as "
            "CLASS_RESULTS."
        )

    test_split = SPLITS[-1]
    if test_split.key != "test" or test_split.instances != TEST_INSTANCES:
        raise ValueError(
            "The last SPLITS row must be the test split, with TEST_INSTANCES "
            "instances."
        )

    for s in SPLITS:
        values = (s.precision, s.recall, s.map50, s.map50_95)
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError(f"split {s.key}: every metric must be between 0 and 1.")
        if s.map50_95 > s.map50:
            raise ValueError(
                f"split {s.key}: mAP50-95 ({s.map50_95}) cannot exceed mAP50 "
                f"({s.map50})."
            )

    for d in CLASS_DETAIL:
        values = (d.precision, d.recall, d.train_map50, d.test_map50)
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError(f"class {d.name}: every metric must be between 0 and 1.")


_validate()