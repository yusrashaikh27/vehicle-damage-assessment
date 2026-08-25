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

3. NEAR-DUPLICATE IMAGES ACROSS SPLITS. A handful of photographs appear in more
   than one split as a re-crop or re-encode. Every one of those inflates the
   score, because the model is being tested on something it trained on. There
   are 50 such clusters - not many, but they cost nothing to remove.

4. THE NON-PROBLEM: FILENAME COLLISIONS. The export contains 10,000 files but
   only 3,969 distinct filename stems, which looks like each photo was
   duplicated 2-3 times. It wasn't. Each merged project had its own numbering,
   so '000002' from one project is a completely different car from '000002' in
   another. Comparing image content shows 97% of same-name files are different
   photographs. Group by filename and you throw away ~6,000 real images for
   nothing.

   The lesson worth keeping: filenames are metadata, and metadata lies. Pixels
   don't. The same applies to counting - see the note on trailing newlines in
   docs/DATASET_NOTES.md.

HOW DUPLICATES ARE ACTUALLY DETECTED
------------------------------------
Two independent signals, because neither is sufficient alone:

  * a 256-bit dHash (16x17 grayscale gradient hash) - sharp, but sensitive to
    cropping, so a genuine re-crop can score as high as ~63;
  * a 512-bin RGB colour histogram intersection - survives crops and rescaling,
    but too blunt on its own (many of these images are a grey panel on a black
    background).

Requiring BOTH (dHash <= 64 and histogram >= 0.88) was calibrated by eye against
sample pairs: it accepts a re-cropped photo of the same wheel, and rejects two
different grey cars that merely share a silhouette. Matched images are unioned
into clusters, and a cluster is treated as one indivisible unit by the splitter,
so no photograph can straddle two splits.

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

def signatures(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """A 256-bit dHash and a 512-bin RGB histogram for every image."""
    dhash = np.zeros((len(records), 256), dtype=np.int8)
    hist = np.zeros((len(records), 512), dtype=np.float32)
    for i, record in enumerate(records):
        image = Image.open(record["image"]).convert("RGB")
        grey = np.asarray(image.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
        dhash[i] = np.where((grey[:, 1:] > grey[:, :-1]).ravel(), 1, -1)
        # 3 bits per channel: coarse enough to survive re-encoding, fine enough
        # to tell a white car from a grey one.
        quantised = np.asarray(image.resize((64, 64), Image.BILINEAR), dtype=np.uint8) >> 5
        counts = np.bincount(
            (quantised[..., 0] * 64 + quantised[..., 1] * 8 + quantised[..., 2]).ravel(),
            minlength=512,
        )
        hist[i] = counts / counts.sum()
    return dhash, hist


def cluster_duplicates(dhash: np.ndarray, hist: np.ndarray) -> list[int]:
    """
    Union-find over confirmed duplicate pairs. Returns a cluster id per image.

    hamming(i, j) = (256 - dot(i, j)) / 2 when bits are stored as +/-1, which
    turns 50 million comparisons into one chunked matrix multiply.
    """
    n = len(dhash)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for start in range(0, n, 1000):
        block = dhash[start:start + 1000].astype(np.int16)
        hamming = (256 - block @ dhash.T.astype(np.int16)) // 2
        for row, j in np.argwhere(hamming <= DHASH_MAX):
            i = start + row
            if i >= j:
                continue
            # second opinion, because dHash alone flags different cars that
            # happen to share a silhouette
            if float(np.minimum(hist[i], hist[j]).sum()) >= HISTOGRAM_MIN:
                union(i, j)

    return [find(i) for i in range(n)]


# --------------------------------------------------------------------------- #
# splitting                                                                    #
# --------------------------------------------------------------------------- #

def stratified_group_split(groups: dict[int, list[int]], records: list[dict]) -> dict[str, list[int]]:
    """
    Deal whole groups into train/valid/test while keeping rare classes present
    in all three.

    Plain random splitting would leave the thin classes unevenly represented -
    'tire_flat' has ~480 instances against 'scratch's ~8,200, so a bad shuffle
    can leave test with too few to compute a meaningful AP. Groups are therefore
    dealt rarest-class-first: the classes that can least afford bad luck get
    placed while every split still has room.
    """
    group_classes = {
        key: {cls for i in members for cls, _ in records[i]["rows"]}
        for key, members in groups.items()
    }
    frequency = Counter(cls for classes in group_classes.values() for cls in classes)
    rarest_first = [cls for cls, _ in sorted(frequency.items(), key=lambda kv: kv[1])]

    rng = random.Random(RANDOM_SEED)
    assignment: dict[str, list[int]] = {name: [] for name in SPLIT_RATIOS}
    unplaced = set(groups)

    def deal(keys: list[int]) -> None:
        """Give each group to whichever split is furthest below its target share."""
        rng.shuffle(keys)
        for key in keys:
            placed = sum(len(v) for v in assignment.values()) or 1
            neediest = max(
                SPLIT_RATIOS,
                key=lambda s: SPLIT_RATIOS[s] - len(assignment[s]) / placed,
            )
            assignment[neediest].append(key)
            unplaced.discard(key)

    for cls in rarest_first:
        deal([k for k in sorted(unplaced) if cls in group_classes[k]])
    deal(sorted(unplaced))  # groups whose images carry no annotations at all

    return assignment


# --------------------------------------------------------------------------- #
# writing                                                                      #
# --------------------------------------------------------------------------- #

def write_split(
    split: str,
    group_keys: list[int],
    groups: dict[int, list[int]],
    records: list[dict],
    class_mapping: dict[int, int],
) -> tuple[Counter, int, int]:
    """
    Copy one image per duplicate cluster into dataset/, rewriting class ids.

    Keeping every member of a cluster would just feed the model the same
    photograph twice, so the least-compressed copy (largest file) wins.
    """
    image_out = OUTPUT_ROOT / "images" / split
    label_out = OUTPUT_ROOT / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    written = background = 0
    for key in sorted(group_keys):
        record = max((records[i] for i in groups[key]), key=lambda r: r["bytes"])
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

    return counts, written, background


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
    stems = {r["image"].stem.split("_jpg.rf.")[0] for r in records}
    print(f"  {len(records)} labelled images, {len(stems)} distinct filename stems")
    print("  (stems collide because the export merges several annotation projects;")
    print("   duplicates are found from pixels below, not from names)")

    print("\nHashing images and clustering duplicates ...")
    dhash, hist = signatures(records)
    cluster_ids = cluster_duplicates(dhash, hist)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, cluster in enumerate(cluster_ids):
        groups[cluster].append(index)
    clustered = sum(len(v) for v in groups.values() if len(v) > 1)
    print(f"  {len(groups)} clusters from {len(records)} images "
          f"({clustered} images sit in a cluster of 2 or more)")
    spanning = sum(1 for v in groups.values() if len({records[i]['origin'] for i in v}) > 1)
    print(f"  {spanning} of those clusters straddled the export's own train/valid/test "
          f"boundary - that was leakage, and it stops here")

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    print("\nSplitting by cluster (rarest class first) ...")
    assignment = stratified_group_split(groups, records)

    print("\nWriting dataset/ ...")
    per_split = {}
    for split, keys in assignment.items():
        per_split[split] = write_split(split, keys, groups, records, class_mapping)

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
                  f"{len(shared)} shared duplicate clusters")

    print("\nInstances per class")
    print(f"  {'class':<16}" + "".join(f"{s:>8}" for s in SPLIT_RATIOS) + f"{'total':>8}")
    for class_id, name in enumerate(target_names):
        row = [per_split[s][0][class_id] for s in SPLIT_RATIOS]
        print(f"  {name:<16}" + "".join(f"{n:>8}" for n in row) + f"{sum(row):>8}")
    images = [per_split[s][1] for s in SPLIT_RATIOS]
    print(f"  {'images':<16}" + "".join(f"{n:>8}" for n in images) + f"{sum(images):>8}")
    empties = [per_split[s][2] for s in SPLIT_RATIOS]
    print(f"  {'(no damage)':<16}" + "".join(f"{n:>8}" for n in empties) + f"{sum(empties):>8}")

    print(f"\nWrote {yaml_path.relative_to(PROJECT_ROOT)}")
    if not ok:
        print("Leakage check FAILED - do not train on this.")
        return 1
    print("No duplicate cluster spans two splits. Safe to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
