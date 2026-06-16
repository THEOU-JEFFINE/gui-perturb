# gui-perturb

A lightweight pipeline for generating **visual perturbations of GUI screenshots**
to evaluate the robustness of Vision-Language-Action (VLA) GUI agents (e.g.
ShowUI, SpiritSight) on grounding benchmarks such as **ScreenSpot-V2**.

Standard GUI grounding benchmarks evaluate agents on clean, well-formed
screenshots. Real interfaces contain pop-ups, injected text, theme changes, and
low-resolution rendering. This tool produces four families of perturbation on
top of an existing benchmark image **without occluding the target element**, so
the change in grounding accuracy can be attributed to the perturbation itself.

## Perturbation families

| Name         | What it does                                              | Failure mode probed            |
|--------------|----------------------------------------------------------|--------------------------------|
| `overlay`    | Pastes a synthetic cookie-banner / modal in free space   | Visual distractor / clutter    |
| `injection`  | Renders adversarial text near (not on) the target        | Visual prompt injection        |
| `theme`      | Dark-mode / high-contrast / grayscale color transform    | Distribution shift             |
| `resolution` | Downsample → upsample to simulate a low-DPI display       | Perceptual degradation         |

**Design constraint:** a perturbation never covers the target element. Overlay
and injection placement search for a region that does not intersect the
(padded) ground-truth box; if none is found the sample is returned unperturbed
and can be logged/skipped. State this constraint when reporting results.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from PIL import Image
from gui_perturb import BBox, PERTURBATIONS

img = Image.open("screenshot.png")
W, H = img.size

# Build a normalized bbox. BE EXPLICIT about the source format:
#   Voxel51 mirror  -> fmt="xywh"
#   OS-Copilot mirror -> fmt="xyxy"
box = BBox.from_raw([0.50, 0.42, 0.10, 0.05], W, H, fmt="xywh", normalized=True)

for name, fn in PERTURBATIONS.items():
    out_img, out_box = fn(img, box, seed=0)
    out_img.save(f"out_{name}.png")
```

## ⚠️ Bounding-box format — read this

The single most common silent bug in GUI-grounding experiments is mixing up
bbox conventions. Mirrors of ScreenSpot-V2 disagree:

- `xyxy` = `[left, top, right, bottom]`   (OS-Copilot)
- `xywh` = `[left, top, width, height]`    (Voxel51)

`BBox.from_raw(..., fmt=..., normalized=...)` forces you to declare both, and
converts everything to an internal **absolute-pixel xyxy** form. The accuracy
metric (`BBox.contains(x, y)`) operates on that single canonical form.

## Dataset

Recommended source (screenshots + annotations bundled):
`HongxinLi/ScreenSpot_v2` on Hugging Face. Verify the bbox format of whichever
mirror you use against a couple of samples before running the full evaluation.

## Status / roadmap

- [x] Four perturbation families
- [x] Format-safe bbox handling + hit-test metric
- [ ] Batch runner over a ScreenSpot-V2 subset
- [ ] Model adapters (ShowUI, SpiritSight) returning normalized (x, y)
- [ ] Mitigation-prompt evaluation harness

## License

MIT
