"""Generate a tiny FAKE mobile-GUI dataset so you can test the whole pipeline
(steps -> ShareGPT -> ready for training) WITHOUT downloading anything.

It draws simple phone-like screenshots with PIL and writes matching steps.jsonl.
The coordinates it produces are already in canonical 0-1000 space.

Run:  python scripts/make_synthetic_demo.py
Then: python src/make_sharegpt.py data/synthetic/steps.jsonl data/synthetic/sharegpt.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PIL import Image, ImageDraw

from action_space import Action
from coords import GRID, norm_from_pixel, box_from_pixel
from schema import Step, save_steps

W, H = 1080, 2400  # a typical phone resolution, in pixels
OUT_DIR = "data/synthetic"
IMG_DIR = os.path.join(OUT_DIR, "imgs")


def norm_box_tuple(px_box):
    b = box_from_pixel(*px_box, W, H)
    return (b.x1, b.y1, b.x2, b.y2)


def center_norm(px_box):
    cx, cy = (px_box[0] + px_box[2]) // 2, (px_box[1] + px_box[3]) // 2
    return norm_from_pixel(cx, cy, W, H)


def draw_screen(path, title, buttons, highlight=None):
    """buttons: list of (label, (x1,y1,x2,y2) in pixels). highlight: index to box."""
    img = Image.new("RGB", (W, H), (245, 245, 248))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 160], fill=(60, 90, 200))      # top bar
    d.text((40, 60), title, fill=(255, 255, 255))
    for i, (label, (x1, y1, x2, y2)) in enumerate(buttons):
        fill = (210, 230, 255) if i == highlight else (255, 255, 255)
        d.rectangle([x1, y1, x2, y2], fill=fill, outline=(150, 150, 150), width=3)
        d.text((x1 + 24, (y1 + y2) // 2 - 8), label, fill=(20, 20, 20))
    img.save(path)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    steps = []

    # One fake 3-step episode: open Settings -> tap Wi-Fi -> toggle on -> done.
    goal = "Turn on Wi-Fi"

    # step 0: home screen, target = open Settings (modeled as open_app)
    p0 = os.path.join(IMG_DIR, "ep1_s0.png")
    draw_screen(p0, "Home", [("Settings", (100, 600, 980, 760)),
                             ("Camera", (100, 820, 980, 980))], highlight=0)
    steps.append(Step("ep1", 0, goal, Action("open_app", app="Settings"),
                      p0, W, H, history=[], thought="Open the Settings app."))

    # step 1: settings list, target = click the Network row
    p1 = os.path.join(IMG_DIR, "ep1_s1.png")
    net_box_px = (100, 600, 980, 760)
    draw_screen(p1, "Settings", [("Network & internet", net_box_px),
                                 ("Display", (100, 820, 980, 980))], highlight=0)
    nx, ny = center_norm(net_box_px)
    steps.append(Step("ep1", 1, goal, Action("click", x=nx, y=ny),
                      p1, W, H, history=[Action("open_app", app="Settings")],
                      thought="Tap Network & internet.",
                      gt_box=norm_box_tuple(net_box_px)))

    # step 2: wifi screen, target = toggle, then we're done
    p2 = os.path.join(IMG_DIR, "ep1_s2.png")
    wifi_box_px = (760, 600, 940, 720)
    draw_screen(p2, "Network & internet", [("Wi-Fi          [OFF]", (100, 600, 980, 760))],
                highlight=0)
    wx, wy = center_norm(wifi_box_px)
    steps.append(Step("ep1", 2, goal, Action("click", x=wx, y=wy),
                      p2, W, H,
                      history=[Action("open_app", app="Settings"),
                               Action("click", x=nx, y=ny)],
                      thought="Tap the Wi-Fi toggle to turn it on.",
                      gt_box=norm_box_tuple(wifi_box_px)))

    save_steps(steps, os.path.join(OUT_DIR, "steps.jsonl"))
    print(f"wrote {len(steps)} steps + {len(steps)} screenshots under {OUT_DIR}/")
    print("next: python src/make_sharegpt.py data/synthetic/steps.jsonl data/synthetic/sharegpt.json")


if __name__ == "__main__":
    main()
