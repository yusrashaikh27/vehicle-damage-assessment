"""
Turn the raw Roboflow CarDD export into a clean, honest training set.

Run from the project root:

    python3 tools/prepare_dataset.py

Reads  : Detection.v1-cardd.yolov11/{train,valid,test}/{images,labels}
Writes : dataset/{images,labels}/{train,valid,test} + dataset/data.yaml
Needs  : numpy + Pillow (both arrive with ultralytics)

WHY THIS SCRIPT EXISTS
----------------------
Three real problems with the export, plus one that looks like a problem and isn't.

1. INCONSISTENT, OVERLAPPING LABELS. The export is several re-annotation
   projects merged together. Where two projects annotated the same photograph
   they disagree about the class roughly 8% of the time - one marks a panel
   'crack', another marks the same panel 'rub' + 'scratch'. Some distinctions in
   the 10-class taxonomy are not reliably separable even by the people who drew
   them: 'rub' versus 'scratch', and 'no part' versus 'dislocated part'. A
   detector trained on contradictory labels spends its capacity learning the
   annotators' disagreement instead of learning damage.

2. A TEST SPLIT TOO SMALL FOR THE RARE CLASSES. The export splits 70/20/10, so
   only 39 of the 487 'tire flat' instances land in test. An AP computed from 39
   instances swings wildly and means very little. Re-splitting 70/15/15, with
   groups dealt rarest-class-first, roughly doubles that.

3. NEAR-DUPLICATE IMAGES ACROSS SPLITS, AND THIS IS THE BIG ONE. The export is
   ~60% pre-baked Roboflow augmentation: each source photograph appears 2-3
   times as a rotation or shear, which pads the frame with pure black. 10,000
   files hold only 3,963 distinct photographs.

   Split those naively and a rotation of a training photograph lands in test.
   Measured on the previous version of this script's output: 81.5% of the test
   split and 80.8% of validation were augmented variants of a TRAIN photograph,
   and 48% of augmentation families straddled a split boundary. An mAP measured
   on that is not inflated, it is meaningless.

   Fixed in two parts, both below: group by augmentation family as well as by
   pixel similarity, and keep only the least-padded member of each family, since
   a black-cornered rotation never occurs at inference time.

4. A TRAP: EARLIER VERSIONS OF THIS FILE SAID FILENAME GROUPING WAS WRONG.
   The claim was that 97-98% of same-stem files are different photographs, on the
   grounds that each merged project numbered from 1 independently, so '000002'
   from one project is a different car from '000002' in another.

   That claim is false, and the reason it was believed is worth keeping. It came
   from comparing raw colour histograms and assuming "a rotation barely changes
   the colour histogram". A rotation adds up to 28% pure-black padding, which
   rewrites the histogram and drags a same-photo pair BELOW the 0.88 acceptance
   threshold. Same photograph, scored as two.

   Mask the near-black pixels, renormalise, and the two populations separate
   almost perfectly (measured 2026-08-30):

     within-family pairs      median 0.87, none below 0.60
     different-family pairs   median 0.37, none above 0.78   (control, n=250)

   Confirmed by eye on the four hardest ambiguous-band families: every one is a
   single photograph rotated, one of them sharing a camera timestamp across all
   three variants.

   The lesson is not "filenames lie" but "check what your similarity metric is
   actually invariant to". Here the metric was not invariant to the exact
   transformation being searched for.

HOW IMAGES ARE GROUPED SO NOTHING LEAKS
---------------------------------------
Two grouping signals, unioned, because each catches what the other misses.

A. AUGMENTATION FAMILY - the filename stem before '_jpg.rf.'. Roboflow names every
   output of one source photograph with that photograph's stem, so the family is
   an exact, free record of which files came from the same original. This catches
   the ~60% of the export that is pre-baked augmentation.

B. PIXEL SIMILARITY - for the same photograph submitted to two annotation
   projects under different names, where the filenames give nothing away:

     * a 256-bit dHash (16x17 grayscale gradient hash) - sharp, but sensitive to
       cropping, so a genuine re-crop can score as high as ~63;
     * a 512-bin RGB colour histogram intersection - survives crops and
       rescaling, but too blunt on its own (many of these images are a grey panel
       on a black background).

   Requiring BOTH (dHash <= 64 and histogram >= 0.88) accepts a re-cropped photo
   of the same wheel and rejects two different grey cars that merely share a
   silhouette. Note this pair of signals does NOT reliably catch rotations, which
   is precisely why (A) exists.

Both are unioned into clusters, and a cluster is one indivisible unit to the
splitter, so no photograph can straddle two splits in any form.

WHICH MEMBER OF A CLUSTER SURVIVES
----------------------------------
One image per cluster is written, and it is the LEAST PADDED member - the one
with the smallest fraction of pure-black pixels. That member is the original
photograph, or near enough (a flip or a brightness shift leaves no padding and is
a perfectly valid car photo).

Discarding the padded rotations rather than keeping them for training is
deliberate. They add no information - they are the same photographs - and they
carry an artefact that never occurs at inference: black wedges in the corners.
Ultralytics already augments on the fly every epoch, with rotation, scaling, HSV
shifts, flips and mosaic, and it letterboxes with grey (114,114,114) rather than
black. So on-the-fly augmentation is strictly better than the pre-baked kind, and
the smaller set means each epoch is ~2.5x faster, which buys more epochs in the
same wall-clock hour on a free Colab T4.

HOW THE CLASSES ARE MERGED
--------------------------
Not by how the damage looks, but by the repair action it implies - because the
cost estimator downstream is driven by repair action, not by visual label.
'rub' and 'scratch' both mean re-paint. 'no part', 'dislocated part' and 'crash'
all mean the panel is refitted or replaced. Ten labels collapse to seven, which
removes exactly the distinctions the annotators disagreed about and lifts the
smallest class from 229 instances to 483.

If you want to argue the other side in a viva: the 10-class taxonomy is more
granular and every class is trainable (the smallest, 'crash', has 229
instances). Edit MERGE_GROUPS to keep them separate and everything downstream
follows, since the class names are read from the generated data.yaml.
"""

from __future__ import annotations

import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT_ROOT / "Detection.v1-cardd.yolov11"
OUTPUT_ROOT = PROJECT_ROOT / "dataset"

# Deterministic, so re-running reproduces the identical split. A split you
# cannot reproduce is a split you cannot defend when the numbers are questioned.
RANDOM_SEED = 42

SPLIT_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}

# Duplicate-detection thresholds - see the module docstring for calibration.
DHASH_MAX = 64
HISTOGRAM_MIN = 0.88

# Roboflow names every augmented output of one source photograph as
# '<stem>_jpg.rf.<32-hex>.jpg', so everything before this marker identifies the
# source photograph. That makes the augmentation family free to recover from the
# filename - see the module docstring for the evidence that these really are the
# same photograph, and for why an earlier version of this file wrongly denied it.
FAMILY_MARKER = "_jpg.rf."

# A pixel is "padding" if every channel is at or below this. Rotation and shear
# pad with exact black, so the bar is deliberately low: JPEG ringing around the
# padding edge lifts a few pixels off zero, but genuine dark paint and shadow sit
# well above it.
PADDING_LEVEL = 8

# Above this fraction of padding pixels an image is *reported* as augmented, and
# is barred from valid/test. Only the printed diagnostics depend on the exact
# value; which member of a cluster survives does not, since members are ranked
# against each other rather than against a threshold.
PADDING_REPORT_MIN = 0.02

# Below this many test instances a per-class AP is too noisy to quote as a point
# value. The export's own 70/20/10 split left 'tire flat' with 39 test instances,
# which was one of the reasons for re-splitting; this is the bar that judgement
# implies, and the script warns when a class falls under it.
MIN_TEST_INSTANCES = 40

# Class ids as they appear in the raw export's data.yaml, with the instance count
# each one actually has (counted per file - see docs/DATASET_NOTES.md for why
# counting with `cat labels/*.txt` gives wrong answers on this dataset).
SOURCE_NAMES = [
    "crack",            # 0   2100
    "crash",            # 1    229
    "dent",             # 2   5347
    "dislocated part",  # 3   1571
    "glass shatter",    # 4   1377
    "lamp broken",      # 5   1355
    "no part",          # 6    870
    "rub",              # 7   1087
    "scratch",          # 8   7228
    "tire flat",        # 9    487
]

# Grouped by the repair action each damage implies. Edit this list if you want a
# different taxonomy - everything downstream reads the names out of data.yaml.
MERGE_GROUPS: list[tuple[str, list[str]]] = [
    ("scratch",         ["scratch", "rub"]),                          # re-paint
    ("dent",            ["dent"]),                                    # pull out + re-paint
    ("crack",           ["crack"]),                                   # repair or replace part
    ("lamp_broken",     ["lamp broken"]),                             # replace lamp unit
    ("dislocated_part", ["dislocated part", "no part", "crash"]),     # refit or replace panel
    ("glass_shatter",   ["glass shatter"]),                           # replace glass
    ("tire_flat",       ["tire flat"]),                               # repair or replace tyre
]


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #

def build_class_mapping() -> tuple[dict[int, int], list[str]]:
    """Map every source class id to its merged id, and return the new name list."""
    target_names = [name for name, _ in MERGE_GROUPS]
    mapping: dict[int, int] = {}
    for target_id, (_, members) in enumerate(MERGE_GROUPS):
        for member in members:
            mapping[SOURCE_NAMES.index(member)] = target_id
    missing = set(range(len(SOURCE_NAMES))) - set(mapping)
    if missing:
        raise SystemExit(
            "MERGE_GROUPS does not account for source classes: "
            + ", ".join(SOURCE_NAMES[i] for i in sorted(missing))
        )
    return mapping, target_names


def read_label(path: Path) -> list[tuple[int, str]]:
    """Return [(class_id, 'x1 y1 x2 y2 ...'), ...] for one YOLO-segmentation file."""
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:  # a class id plus at least 3 xy pairs (a triangle)
            continue
        rows.append((int(parts[0]), " ".join(parts[1:])))
    return rows


def load_records() -> list[dict]:
    """Every labelled image in the export, regardless of which split it sits in."""
    records = []
    for split in ("train", "valid", "test"):
        image_dir = SOURCE_ROOT / split / "images"
        label_dir = SOURCE_ROOT / split / "labels"
        if not image_dir.is_dir():
            raise SystemExit(f"Missing {image_dir} - is the Roboflow export unzipped?")
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue  # an image with no label file teaches the model nothing
            records.append({
                "image": image_path,
                "rows": read_label(label_path),
                "bytes": image_path.stat().st_size,
                "origin": split,
            })
    return records


# --------------------------------------------------------------------------- #
# duplicate detection                                                          #
# --------------------------------------------------------------------------- #

def signatures(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A 256-bit dHash, a 512-bin RGB histogram, and a padding score per image.

    All three come from one decode of each image, because decoding 10,000 JPEGs is
    the slowest thing this script does.

    The padding score is the fraction of near-black pixels. Rotation and shear pad
    the frame with exact black, so a padded copy scores 0.15-0.30 while an
    untouched photograph scores near zero. It is used to decide which member of a
    duplicate cluster is the original.
    """
    dhash = np.zeros((len(records), 256), dtype=np.int8)
    hist = np.zeros((len(records), 512), dtype=np.float32)
    padding = np.zeros(len(records), dtype=np.float32)
    for i, record in enumerate(records):
        image = Image.open(record["image"]).convert("RGB")
        grey = np.asarray(image.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
        dhash[i] = np.where((grey[:, 1:] > grey[:, :-1]).ravel(), 1, -1)
        # 3 bits per channel: coarse enough to survive re-encoding, fine enough
        # to tell a white car from a grey one.
        small = np.asarray(image.resize((64, 64), Image.BILINEAR), dtype=np.uint8)
        quantised = small >> 5
        counts = np.bincount(
            (quantised[..., 0] * 64 + quantised[..., 1] * 8 + quantised[..., 2]).ravel(),
            minlength=512,
        )
        hist[i] = counts / counts.sum()
        padding[i] = float((small.max(axis=2) <= PADDING_LEVEL).mean())
    return dhash, hist, padding



def cluster_duplicates(
    dhash: np.ndarray, hist: np.ndarray, padding: np.ndarray, records: list[dict]
) -> tuple[list[int], int, int, int, int]:
    """
    Union-find over two kinds of match, applied in this order for a reason.

    Signal A, augmentation family: same filename stem before '_jpg.rf.'. This
    catches the padded rotations, which are ~60% of the export and which the pixel
    signals below miss entirely - rotating an image moves its dHash to a distance
    of ~100, far past DHASH_MAX.

    Signal B, pixel similarity: dHash within DHASH_MAX *and* histogram
    intersection at or above HISTOGRAM_MIN. This catches the same photograph
    submitted to two annotation projects under unrelated names, where the
    filenames give nothing away.

    Signal B runs only on one representative image per family - the least-padded
    member - and that is not merely an optimisation. Padded rotations are largely
    black, so their gradient hashes all resemble one another, and comparing all
    10,000 images produces a candidate list big enough to stall the run while
    finding nothing that signal A has not already grouped. Comparing ~4,000
    unpadded photographs is 6x fewer comparisons and is also the regime the
    DHASH_MAX/HISTOGRAM_MIN pair was calibrated for in the first place.

    hamming(i, j) = (256 - dot(i, j)) / 2 when bits are stored as +/-1, which
    turns the comparison into one chunked matrix multiply.

    Returns (cluster id per image, family unions, pixel unions, family count,
    candidate pairs considered by signal B).
    """
    n = len(dhash)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[max(ra, rb)] = min(ra, rb)
        return True

    # --- Signal A: augmentation family ------------------------------------- #
    by_family: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        by_family[record["image"].stem.split(FAMILY_MARKER)[0]].append(i)
    family_unions = 0
    for members in by_family.values():
        for other in members[1:]:
            family_unions += union(members[0], other)

    # --- Signal B: pixel similarity, representatives only ------------------- #
    reps = np.array(sorted(
        min(members, key=lambda i: (round(float(padding[i]), 4), -records[i]["bytes"]))
        for members in by_family.values()
    ))
    rep_dhash, rep_hist = dhash[reps], hist[reps]
    m = len(reps)

    candidates = []
    for start in range(0, m, 512):
        block = rep_dhash[start:start + 512].astype(np.int16)
        hamming = (256 - block @ rep_dhash.T.astype(np.int16)) // 2
        rows, cols = np.nonzero(hamming <= DHASH_MAX)
        rows = rows + start
        upper = rows < cols  # each pair once, and never an image against itself
        if upper.any():
            candidates.append(np.stack([rows[upper], cols[upper]], axis=1))
    pairs = np.concatenate(candidates) if candidates else np.zeros((0, 2), dtype=int)

    # The histogram second opinion, vectorised in chunks. Done pair-by-pair in
    # Python this is the slowest line in the script; done as one array it is
    # bounded by memory, hence the chunking.
    pixel_unions = 0
    for start in range(0, len(pairs), 20_000):
        chunk = pairs[start:start + 20_000]
        agree = np.minimum(rep_hist[chunk[:, 0]], rep_hist[chunk[:, 1]]).sum(axis=1)
        for i, j in chunk[agree >= HISTOGRAM_MIN]:
            pixel_unions += union(int(reps[i]), int(reps[j]))

    return [find(i) for i in range(n)], family_unions, pixel_unions, len(by_family), len(pairs)




# --------------------------------------------------------------------------- #
# splitting                                                                    #
# --------------------------------------------------------------------------- #

def choose_representatives(
    groups: dict[int, list[int]], records: list[dict], padding: np.ndarray
) -> dict[int, int]:
    """
    Pick the one image that represents each cluster: the least-padded member.

    Every member of a cluster is the same photograph, so keeping more than one
    would just show the model the same picture twice. The least-padded member is
    the original, or a flip or brightness shift of it, which is an equally valid
    photograph. Ties break on the largest file, i.e. the least-recompressed copy.

    Returns {cluster key: index into records}.
    """
    return {
        key: min(members, key=lambda i: (round(float(padding[i]), 4), -records[i]["bytes"]))
        for key, members in groups.items()
    }


def stratified_group_split(
    representatives: dict[int, int], records: list[dict], train_only: set[int]
) -> dict[str, list[int]]:
    """
    Deal whole clusters into train/valid/test while keeping rare classes present
    in all three.

    Two constraints, in priority order.

    First, clusters in `train_only` go to train and nowhere else. These are
    photographs whose every surviving copy is a padded rotation, because Roboflow
    augmented only the export's train split, so a photograph that lived solely
    there has no clean copy. A padded frame is fine to learn from but must not be
    measured on: black corner wedges never occur when a user uploads a photo, so
    scoring on them measures the wrong distribution. Valid and test therefore end
    up entirely real photographs.

    Second, the rest are dealt rarest-class-first, and each cluster goes to
    whichever split is furthest below its target share *of that class's
    instances*.

    Balancing on instance counts rather than on image counts matters, and getting
    it wrong is not hypothetical: an earlier version of this function balanced on
    image count, and constraint one then broke it. Pinning 1,468 padded clusters
    to train before dealing made train start out looking 53% full, so every split
    decision taken early sent its cluster to valid or test to even the image
    counts up - and the rarest classes are dealt first, so they are exactly what
    got sent away. 'tire_flat' ended up with 87 instances in train against 124
    across valid and test. That is the wrong way round. A model that barely sees
    flat tyres scores near zero on them however many test instances exist, so
    starving train is the worse of the two failures. Balancing per class keeps
    each class near 70/15/15 in its own right, whatever constraint one did to the
    image counts.

    Balancing uses the class set of each cluster's *representative*, not of every
    member, because the representative is the only image that gets written.
    """
    rows = {key: records[index]["rows"] for key, index in representatives.items()}
    group_classes = {key: {cls for cls, _ in r} for key, r in rows.items()}
    class_total = Counter(cls for r in rows.values() for cls, _ in r)
    rarest_first = [cls for cls, _ in sorted(class_total.items(), key=lambda kv: kv[1])]

    rng = random.Random(RANDOM_SEED)
    assignment: dict[str, list[int]] = {name: [] for name in SPLIT_RATIOS}
    placed: dict[str, Counter] = {name: Counter() for name in SPLIT_RATIOS}

    def place(key: int, split: str) -> None:
        assignment[split].append(key)
        for cls, _ in rows[key]:
            placed[split][cls] += 1

    # Constraint one, applied before anything is dealt.
    for key in sorted(train_only):
        place(key, "train")
    unplaced = set(representatives) - train_only

    def deficit(split: str, cls: int | None) -> float:
        """How far below target this split is, as a fraction of its target."""
        if cls is None:  # no annotations to balance on; fall back to image count
            target = SPLIT_RATIOS[split] * len(representatives)
            return (target - len(assignment[split])) / max(target, 1.0)
        target = SPLIT_RATIOS[split] * class_total[cls]
        return (target - placed[split][cls]) / max(target, 1.0)

    def deal(keys: list[int], cls: int | None) -> None:
        rng.shuffle(keys)
        for key in keys:
            place(key, max(SPLIT_RATIOS, key=lambda s: deficit(s, cls)))
            unplaced.discard(key)

    for cls in rarest_first:
        deal([k for k in sorted(unplaced) if cls in group_classes[k]], cls)
    deal(sorted(unplaced), None)  # clusters whose images carry no annotations at all

    return assignment




# --------------------------------------------------------------------------- #
# writing                                                                      #
# --------------------------------------------------------------------------- #

def write_split(
    split: str,
    group_keys: list[int],
    representatives: dict[int, int],
    records: list[dict],
    class_mapping: dict[int, int],
    padding: np.ndarray,
) -> dict:
    """
    Copy each cluster's representative image into dataset/, rewriting class ids.

    Returns per-split counts, including how many written images still carry
    padding - which should be near zero for valid and test, and is the check that
    the augmentation copies really were excluded.
    """
    image_out = OUTPUT_ROOT / "images" / split
    label_out = OUTPUT_ROOT / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    written = background = padded = 0
    for key in sorted(group_keys):
        index = representatives[key]
        record = records[index]
        stem = record["image"].stem
        shutil.copy2(record["image"], image_out / f"{stem}{record['image'].suffix.lower()}")

        lines = []
        for cls, coords in record["rows"]:
            merged = class_mapping[cls]
            counts[merged] += 1
            lines.append(f"{merged} {coords}")
        (label_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        written += 1
        background += not lines
        padded += float(padding[index]) > PADDING_REPORT_MIN

    return {"counts": counts, "images": written, "background": background, "padded": padded}


def audit_families(assignment: dict[str, list[int]], representatives: dict[int, int],
                   records: list[dict]) -> dict[str, int]:
    """
    The check that matters: does any augmentation family have images in two splits?

    Splitting by cluster should make this structurally impossible, but 'should' is
    not evidence. This recomputes families from the filenames that were actually
    written and counts how many span a split boundary - it must be zero.
    """
    family_splits: dict[str, set[str]] = defaultdict(set)
    for split, keys in assignment.items():
        for key in keys:
            stem = records[representatives[key]]["image"].stem
            family_splits[stem.split(FAMILY_MARKER)[0]].add(split)
    return {
        "families": len(family_splits),
        "spanning": sum(1 for splits in family_splits.values() if len(splits) > 1),
    }



def write_data_yaml(target_names: list[str]) -> Path:
    """
    Ultralytics resolves train/val/test relative to `path`, and finds labels by
    swapping '/images/' for '/labels/' - hence this directory layout.
    """
    path = OUTPUT_ROOT / "data.yaml"
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(target_names))
    path.write_text(
        "# Generated by tools/prepare_dataset.py - do not hand-edit.\n"
        "# Duplicate-free splits, merged to 7 repair-action classes.\n"
        "path: .\n"
        "train: images/train\n"
        "val: images/valid\n"
        "test: images/test\n"
        f"nc: {len(target_names)}\n"
        "names:\n"
        f"{names}\n"
    )
    return path


# --------------------------------------------------------------------------- #

def main() -> int:
    class_mapping, target_names = build_class_mapping()

    print("Reading the raw export ...")
    records = load_records()
    stems = {r["image"].stem.split(FAMILY_MARKER)[0] for r in records}
    print(f"  {len(records)} labelled images, {len(stems)} distinct augmentation families")
    print(f"  so roughly {1 - len(stems) / len(records):.0%} of the export is augmentation copies")

    print("\nHashing images, scoring padding, clustering ...")
    dhash, hist, padding = signatures(records)
    heavily_padded = int((padding > PADDING_REPORT_MIN).sum())
    print(f"  {heavily_padded} of {len(records)} images carry black padding "
          f"({heavily_padded / len(records):.1%}) - rotated or sheared copies")

    cluster_ids, family_unions, pixel_unions, n_families, n_pairs = cluster_duplicates(
        dhash, hist, padding, records
    )
    groups: dict[int, list[int]] = defaultdict(list)
    for index, cluster in enumerate(cluster_ids):
        groups[cluster].append(index)
    clustered = sum(len(v) for v in groups.values() if len(v) > 1)
    print(f"  {family_unions} unions from filename families ({n_families} families)")
    print(f"  {pixel_unions} more from pixel similarity, from {n_pairs} candidate pairs "
          f"compared between family representatives")
    print(f"  {len(groups)} clusters from {len(records)} images "
          f"({clustered} images sit in a cluster of 2 or more)")
    spanning = sum(1 for v in groups.values() if len({records[i]['origin'] for i in v}) > 1)
    print(f"  {spanning} of those clusters straddled the export's own train/valid/test "
          f"boundary - that was leakage, and it stops here")

    representatives = choose_representatives(groups, records, padding)
    train_only = {
        key for key, index in representatives.items()
        if float(padding[index]) > PADDING_REPORT_MIN
    }
    print(f"\nOne image kept per cluster, the least-padded member:")
    print(f"  {len(train_only)} of {len(representatives)} kept images still carry padding")
    print(f"  (photographs whose every copy was augmented - these are pinned to train,")
    print(f"   so that valid and test contain only real, unpadded photographs)")

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    print("\nSplitting by cluster (padded pinned to train, then rarest class first) ...")
    assignment = stratified_group_split(representatives, records, train_only)

    print("\nWriting dataset/ ...")
    per_split = {}
    for split, keys in assignment.items():
        per_split[split] = write_split(
            split, keys, representatives, records, class_mapping, padding
        )

    yaml_path = write_data_yaml(target_names)

    print("\nLeakage check")
    ok = True
    names = list(assignment)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(assignment[a]) & set(assignment[b])
            if shared:
                ok = False
            print(f"  {'OK  ' if not shared else 'FAIL'} {a} vs {b}: "
                  f"{len(shared)} shared clusters")

    audit = audit_families(assignment, representatives, records)
    if audit["spanning"]:
        ok = False
    print(f"  {'OK  ' if not audit['spanning'] else 'FAIL'} "
          f"{audit['spanning']} of {audit['families']} augmentation families span two splits")

    padded_eval = per_split["valid"]["padded"] + per_split["test"]["padded"]
    if padded_eval:
        ok = False
    print(f"  {'OK  ' if not padded_eval else 'FAIL'} "
          f"{padded_eval} padded images in valid/test "
          f"(must be 0 - padding never occurs at inference)")

    print("\nInstances per class")
    print(f"  {'class':<16}" + "".join(f"{s:>8}" for s in SPLIT_RATIOS) + f"{'total':>8}")
    thin = []
    for class_id, name in enumerate(target_names):
        row = [per_split[s]["counts"][class_id] for s in SPLIT_RATIOS]
        flag = ""
        if row[list(SPLIT_RATIOS).index("test")] < MIN_TEST_INSTANCES:
            flag = "  <- too thin to trust its AP"
            thin.append(name)
        print(f"  {name:<16}" + "".join(f"{n:>8}" for n in row) + f"{sum(row):>8}{flag}")
    for label, field in (("images", "images"), ("(no damage)", "background"),
                         ("(still padded)", "padded")):
        row = [per_split[s][field] for s in SPLIT_RATIOS]
        print(f"  {label:<16}" + "".join(f"{n:>8}" for n in row) + f"{sum(row):>8}")

    if thin:
        print(f"\n  Note: {', '.join(thin)} have fewer than {MIN_TEST_INSTANCES} test")
        print("  instances. Their per-class AP will swing between runs and should be")
        print("  reported with that caveat, or reported as a range, not a point value.")
        print("  This is a limit of the data, not a bug - there are only so many")
        print("  distinct photographs of a flat tyre in CarDD.")

    try:
        shown = yaml_path.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = yaml_path
    print(f"\nWrote {shown}")
    if not ok:
        print("Leakage check FAILED - do not train on this.")
        return 1
    print("No cluster and no augmentation family spans two splits, and valid/test")
    print("contain only unpadded photographs. Safe to train.")
    return 0



if __name__ == "__main__":
    sys.exit(main())
