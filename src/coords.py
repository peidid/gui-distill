"""Coordinate normalization for mobile GUI agents.

THE CANONICAL SPACE is a 0-1000 integer grid (Qwen2.5-VL's native grounding
convention). Every coordinate that enters the system — teacher output,
AndroidControl pixel labels, emulator taps — is converted to this space exactly
once, here. Models read and write 0-1000; only the emulator boundary converts
back to device pixels.

Why this matters (the week-eating gotcha):
  - AndroidControl labels are in DEVICE PIXELS (e.g. 1080x2400), origin top-left.
  - Qwen2.5-VL emits/consumes 0-1000 normalized ints by convention.
  - AndroidWorld executes taps in screen pixels.
These are three different spaces. Mixing them silently trains a model whose
"clicks" land in the wrong place while loss still looks fine. Convert here, test
here, and never hand-roll a conversion elsewhere.

bbox-scaling note for Qwen2.5-VL fine-tuning: the processor resizes images to a
patch-aligned resolution (min/max pixel bounds), so the *pixel* coords the model
sees are NOT the original image pixels. Keeping labels in resolution-independent
0-1000 space sidesteps this entirely — that is the point of normalizing.
"""

from dataclasses import dataclass

GRID = 1000  # canonical normalized range is [0, GRID]


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def norm_from_pixel(x_px, y_px, img_w, img_h):
    """Device-pixel coords -> canonical 0-1000 ints.

    Rounds to nearest int and clamps into range so an off-by-one edge pixel
    can't produce 1001.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"image size must be positive, got {img_w}x{img_h}")
    xn = _clamp(round(x_px / img_w * GRID), 0, GRID)
    yn = _clamp(round(y_px / img_h * GRID), 0, GRID)
    return xn, yn


def pixel_from_norm(x_norm, y_norm, img_w, img_h):
    """Canonical 0-1000 coords -> device pixels (for executing a tap)."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"image size must be positive, got {img_w}x{img_h}")
    xp = _clamp(round(x_norm / GRID * img_w), 0, img_w - 1)
    yp = _clamp(round(y_norm / GRID * img_h), 0, img_h - 1)
    return xp, yp


@dataclass(frozen=True)
class Box:
    """Bounding box in canonical 0-1000 space."""
    x1: int
    y1: int
    x2: int
    y2: int

    def center(self):
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def box_from_pixel(x1, y1, x2, y2, img_w, img_h):
    a = norm_from_pixel(x1, y1, img_w, img_h)
    b = norm_from_pixel(x2, y2, img_w, img_h)
    return Box(a[0], a[1], b[0], b[1])


# ScreenSpot-style grounding metric: a predicted point is correct if it falls
# inside the ground-truth element box. Used by both grounding eval and as a
# component of offline step accuracy.
def point_in_box(pred_xy, gt_box):
    return gt_box.contains(pred_xy[0], pred_xy[1])


if __name__ == "__main__":
    # Self-tests — run: python3 src/coords.py
    W, H = 1080, 2400

    # round-trip stays close (within 1 pixel of rounding)
    for px, py in [(0, 0), (540, 1200), (1079, 2399), (123, 4567 % H)]:
        nx, ny = norm_from_pixel(px, py, W, H)
        rx, ry = pixel_from_norm(nx, ny, W, H)
        assert abs(rx - px) <= 2 and abs(ry - py) <= 2, (px, py, nx, ny, rx, ry)

    # corners map sensibly
    assert norm_from_pixel(0, 0, W, H) == (0, 0)
    assert norm_from_pixel(W, H, W, H) == (GRID, GRID)
    assert norm_from_pixel(W // 2, H // 2, W, H) == (500, 500)

    # clamping: out-of-range input never escapes [0, GRID]
    assert norm_from_pixel(99999, 99999, W, H) == (GRID, GRID)
    assert norm_from_pixel(-50, -50, W, H) == (0, 0)

    # pixel output stays a valid index
    assert pixel_from_norm(GRID, GRID, W, H) == (W - 1, H - 1)

    # box + point-in-box metric
    b = box_from_pixel(100, 200, 300, 400, W, H)
    assert b.center() == norm_from_pixel(200, 300, W, H)
    assert point_in_box(b.center(), b)
    assert not point_in_box((0, 0), b)

    # bad image size rejected
    try:
        norm_from_pixel(1, 1, 0, 100)
        assert False, "expected ValueError"
    except ValueError:
        pass

    print("coords.py: all self-tests passed")
