"""ScreenSpot-V2 static grounding eval: instruction + screenshot -> one click;
correct iff the click lands in the ground-truth element box.

This is the single-step / perception side of the gap. No emulator, ~1.2k samples.

Run (on a box with `datasets` + a model):
    python src/eval_screenspot.py --backend mlx --limit 200
    python src/eval_screenspot.py --backend hf --adapter out/qwen3b-trackA-lora

Two things vary across HF mirrors — flags expose both:
  --bbox_format {xywh,xyxy}   ScreenSpot's original boxes are [x,y,w,h] (xywh).
  --hf_name <repo>            which mirror to pull.
Verify on a couple of samples that boxes look sane before trusting the number.
"""

import argparse
from collections import defaultdict

from coords import norm_from_pixel
from eval_core import grounding_correct, summarize
from model import load_model
from prompt import SYSTEM, build_prompt_parts


def bbox_to_norm(bbox, fmt, img_w, img_h):
    """Pixel bbox (xywh or xyxy) -> (x1,y1,x2,y2) in canonical 0-1000 space."""
    if fmt == "xywh":
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:  # xyxy
        x1, y1, x2, y2 = bbox
    a = norm_from_pixel(x1, y1, img_w, img_h)
    b = norm_from_pixel(x2, y2, img_w, img_h)
    return (a[0], a[1], b[0], b[1])


def _get(sample, *keys, default=None):
    for k in keys:
        if k in sample and sample[k] is not None:
            return sample[k]
    return default


def run(dataset, model, bbox_format, img_dir=None):
    import os
    from PIL import Image

    results, by_type = [], defaultdict(list)
    for s in dataset:
        instruction = _get(s, "instruction", "task", "text")
        bbox = _get(s, "bbox", "bounding_box")
        img = _get(s, "image", "img")
        # image may be a PIL.Image (datasets) or a path string
        if isinstance(img, str):
            path = os.path.join(img_dir, img) if img_dir else img
            pil = Image.open(path)
        else:
            pil = img
            path = getattr(img, "filename", None) or _save_tmp(img)
        gt = bbox_to_norm(bbox, bbox_format, pil.width, pil.height)

        user = build_prompt_parts(f"Click on: {instruction}")
        raw = model.generate(path, SYSTEM, user)
        ok = grounding_correct(raw, gt)
        results.append(ok)
        by_type[_get(s, "data_type", "type", default="all")].append(ok)
    return results, by_type


_TMP = []
def _save_tmp(pil):
    import tempfile
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pil.save(f.name)
    _TMP.append(f.name)
    return f.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="dummy", choices=["dummy", "mlx", "hf"])
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--hf_name", default="HongxinLi/ScreenSpot_v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bbox_format", default="xywh", choices=["xywh", "xyxy"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.hf_name, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    model = load_model(args.backend, args.model_path, args.adapter)
    results, by_type = run(ds, model, args.bbox_format)
    s = summarize(results)
    print(f"\nScreenSpot-V2 grounding accuracy: {s['accuracy']:.3f}  "
          f"({s['correct']}/{s['n']})")
    for t, rs in sorted(by_type.items()):
        ss = summarize(rs)
        print(f"  {t:10s} {ss['accuracy']:.3f}  ({ss['correct']}/{ss['n']})")


if __name__ == "__main__":
    main()
