"""
ScreenSpot-V2 loader + batch perturbation generator.

Pulls the benchmark from Hugging Face, converts every annotation into our
canonical absolute-pixel BBox, and (optionally) writes out perturbed copies of
each image for every perturbation family.

-----------------------------------------------------------------------------
⚠️  ONE THING YOU MUST CONFIRM IN COLAB BEFORE A FULL RUN
-----------------------------------------------------------------------------
Different ScreenSpot-V2 mirrors store the bbox differently. After loading,
call `inspect_first(ds)` and look at the printed sample. Then set BBOX_FMT and
BBOX_NORMALIZED below to match what you actually see. Getting this wrong makes
every accuracy number meaningless while looking superficially fine.

  HongxinLi / OS-Copilot mirror  -> often xyxy  ([left, top, right, bottom])
  Voxel51 mirror                 -> often xywh  ([left, top, width, height])

Normalized means values are in [0,1]; if you see numbers like 980, 360 they are
absolute pixels (normalized=False).
-----------------------------------------------------------------------------
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from PIL import Image

from .bbox import BBox
from .perturbations import PERTURBATIONS


# ---- SET THESE AFTER inspect_first() IN COLAB ------------------------------ #
BBOX_FMT = "xyxy"          # "xyxy" or "xywh"
BBOX_NORMALIZED = False    # True if bbox values are in [0,1]
# --------------------------------------------------------------------------- #


@dataclass
class Sample:
    """One benchmark item with its image and canonical bbox."""
    image: Image.Image
    instruction: str
    box: BBox
    data_type: str          # "text" or "icon" — keep for per-type analysis
    source: str             # platform: mobile / desktop / web (if available)
    uid: str


def inspect_first(ds) -> None:
    """Print the first sample's fields so you can confirm the bbox format.
    Run this ONCE in Colab before generating anything."""
    ex = ds[0]
    print("Available fields:", list(ex.keys()))
    for k, v in ex.items():
        if k in ("image", "img"):
            print(f"  {k}: <PIL image> size={v.size}")
        else:
            print(f"  {k}: {v!r}")
    print("\n--> Now set BBOX_FMT / BBOX_NORMALIZED at the top of dataset.py "
          "to match the 'bbox' field above.")


def _get_image(ex) -> Image.Image:
    """Datasets vary in the image field name; handle the common ones."""
    for key in ("image", "img", "screenshot"):
        if key in ex and ex[key] is not None:
            return ex[key].convert("RGB")
    raise KeyError(f"No image field found in sample. Keys: {list(ex.keys())}")


def _get_field(ex, *names, default=""):
    for n in names:
        if n in ex and ex[n] is not None:
            return ex[n]
    return default


def to_sample(ex, idx: int) -> Sample:
    """Convert one raw HF dataset row into a canonical Sample."""
    img = _get_image(ex)
    W, H = img.size
    raw = _get_field(ex, "bbox", "bounding_box", "box")
    box = BBox.from_raw(list(raw), W, H, fmt=BBOX_FMT, normalized=BBOX_NORMALIZED)
    return Sample(
        image=img,
        instruction=str(_get_field(ex, "instruction", "prompt")),
        box=box,
        data_type=str(_get_field(ex, "data_type", "element_type", default="unknown")),
        source=str(_get_field(ex, "data_source", "platform", "source", default="unknown")),
        uid=str(_get_field(ex, "id", "uid", default=f"sample_{idx}")),
    )


def sanity_check_box(s: Sample) -> bool:
    """Cheap guard: a valid box must be inside the image and have positive area.
    If many samples fail this, your BBOX_FMT/NORMALIZED is almost certainly wrong."""
    W, H = s.image.size
    b = s.box
    return (0 <= b.x1 < b.x2 <= W + 1) and (0 <= b.y1 < b.y2 <= H + 1) and b.width > 0 and b.height > 0


def generate_perturbed_set(
    ds,
    out_dir: str,
    n: int | None = 200,
    perturbations: list[str] | None = None,
    seed: int = 0,
) -> dict:
    """Run the pipeline over the first `n` samples, writing clean + perturbed
    images and a manifest. Returns summary stats.

    Output layout:
        out_dir/
          clean/<uid>.png
          overlay/<uid>.png
          injection/<uid>.png
          ...
          manifest.csv          (uid, instruction, data_type, source, box, split)
    """
    perturbations = perturbations or list(PERTURBATIONS.keys())
    splits = ["clean"] + perturbations
    for sp in splits:
        os.makedirs(os.path.join(out_dir, sp), exist_ok=True)

    n = len(ds) if n is None else min(n, len(ds))
    manifest_rows = []
    bad_boxes = 0

    for idx in range(n):
        s = to_sample(ds[idx], idx)
        if not sanity_check_box(s):
            bad_boxes += 1
            continue

        # clean copy
        s.image.save(os.path.join(out_dir, "clean", f"{s.uid}.png"))
        # one row per (sample, split); box is identical across these 4 perturbations
        x1, y1, x2, y2 = s.box.as_tuple()
        base = dict(uid=s.uid, instruction=s.instruction.replace("\n", " "),
                    data_type=s.data_type, source=s.source,
                    x1=round(x1, 2), y1=round(y1, 2), x2=round(x2, 2), y2=round(y2, 2))
        manifest_rows.append({**base, "split": "clean"})

        for pname in perturbations:
            fn = PERTURBATIONS[pname]
            out_img, _ = fn(s.image, s.box, seed=seed)
            out_img.save(os.path.join(out_dir, pname, f"{s.uid}.png"))
            manifest_rows.append({**base, "split": pname})

    # write manifest
    import csv
    man_path = os.path.join(out_dir, "manifest.csv")
    with open(man_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    stats = dict(requested=n, written=len(manifest_rows) // len(splits),
                 bad_boxes=bad_boxes, splits=splits, manifest=man_path)
    return stats
