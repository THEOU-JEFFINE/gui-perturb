"""
Bounding-box handling with EXPLICIT format conversion.

This module exists to prevent the single most common silent bug in GUI-grounding
experiments: mixing up bounding-box conventions. Different ScreenSpot-V2 mirrors
store boxes differently:

  - "xyxy"  -> [left, top, right, bottom]            (OS-Copilot mirror)
  - "xywh"  -> [left, top, width, height]            (Voxel51 mirror)

Coordinates may be NORMALIZED ([0,1] fractions of width/height) or ABSOLUTE
(pixel values). We always convert to a single internal representation:

    BBox stored internally as ABSOLUTE pixel xyxy: (x1, y1, x2, y2)

Everything downstream (perturbation placement, hit-testing) uses that one form.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class BBox:
    """Absolute-pixel bounding box in xyxy form."""
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_raw(
        cls,
        raw: list[float],
        img_w: int,
        img_h: int,
        fmt: str = "xywh",
        normalized: bool = True,
    ) -> "BBox":
        """Build an absolute-pixel xyxy BBox from a raw annotation.

        Args:
            raw: the 4 numbers from the dataset.
            img_w, img_h: image dimensions in pixels.
            fmt: "xywh" (left,top,width,height) or "xyxy" (left,top,right,bottom).
            normalized: True if raw values are [0,1] fractions, False if pixels.
        """
        a, b, c, d = raw
        if normalized:
            a, c = a * img_w, c * img_w
            b, d = b * img_h, d * img_h
        if fmt == "xywh":
            x1, y1, x2, y2 = a, b, a + c, b + d
        elif fmt == "xyxy":
            x1, y1, x2, y2 = a, b, c, d
        else:
            raise ValueError(f"Unknown bbox fmt: {fmt!r}. Use 'xywh' or 'xyxy'.")
        return cls(x1, y1, x2, y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def contains(self, x: float, y: float) -> bool:
        """Hit-test: is point (x,y) inside this box? This IS the accuracy metric."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def scaled(self, sx: float, sy: float) -> "BBox":
        """Return a copy scaled by (sx, sy) — needed when an image is resized."""
        return BBox(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)
