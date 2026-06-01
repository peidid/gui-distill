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

from action_space import XY_ACTIONS
from coords import GRID, Box, norm_from_pixel
from eval_core import safe_parse, summarize
from model import load_model
from prompt import SYSTEM, build_prompt_parts


def bbox_to_norm(bbox, fmt, img_w, img_h):
    """Ground-truth bbox -> (x1,y1,x2,y2) in canonical 0-1000 space.

    Two storage conventions appear across ScreenSpot mirrors, auto-detected:
      - normalized [0,1] *fractions* of the image (HongxinLi/ScreenSpot_v2 stores
        these, as xyxy). Just scale by GRID — image size is irrelevant.
      - absolute pixels in `fmt` (xywh original, or xyxy). Divide by image size.
    A box of all-<=1 values is unambiguously the fraction form (a real pixel box
    can't have x2<=1), so we branch on that.
    """
    if max(bbox) <= 1.0:  # normalized fractions, xyxy
        x1, y1, x2, y2 = bbox
        return (round(x1 * GRID), round(y1 * GRID),
                round(x2 * GRID), round(y2 * GRID))
    if fmt == "xywh":
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:  # xyxy
        x1, y1, x2, y2 = bbox
    a = norm_from_pixel(x1, y1, img_w, img_h)
    b = norm_from_pixel(x2, y2, img_w, img_h)
    return (a[0], a[1], b[0], b[1])


def _pred_point_norm(raw, coord_space, img_w, img_h):
    """Parse the model's click and return it in canonical 0-1000 space, or None.

    coord_space='pixel': the model emits ABSOLUTE pixel coords (Qwen2.5-VL's
    native convention) -> normalize by image size. coord_space='norm': the model
    already emits 0-1000 (e.g. a student fine-tuned on normalized labels).
    """
    pred = safe_parse(raw)
    if pred is None or pred.type not in XY_ACTIONS:
        return None
    if coord_space == "pixel":
        return norm_from_pixel(pred.x, pred.y, img_w, img_h)
    return (pred.x, pred.y)


def _get(sample, *keys, default=None):
    for k in keys:
        if k in sample and sample[k] is not None:
            return sample[k]
    return default


def run(dataset, model, bbox_format, coord_space="pixel", img_dir=None,
        stream=True):
    import os
    import sys
    from PIL import Image

    n_total = len(dataset) if hasattr(dataset, "__len__") else None
    results, by_type = [], defaultdict(list)
    for i, s in enumerate(dataset, 1):
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
        pt = _pred_point_norm(raw, coord_space, pil.width, pil.height)
        ok = pt is not None and Box(*gt).contains(*pt)
        results.append(ok)
        by_type[_get(s, "data_type", "type", default="all")].append(ok)
        if stream:
            # Running accuracy to stderr so it shows live even when stdout is
            # piped to `tee`, and never pollutes the final summary on stdout.
            run_acc = sum(results) / len(results)
            denom = f"/{n_total}" if n_total else ""
            mark = "OK " if ok else "miss"
            print(f"[{i}{denom}] acc={run_acc:.3f} ({sum(results)}/{len(results)})"
                  f"  {mark} {_get(s, 'data_type', 'type', default='all')}",
                  file=sys.stderr, flush=True)
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
    ap.add_argument("--coord_space", default="pixel", choices=["pixel", "norm"],
                    help="how to read the MODEL's click. 'pixel': absolute pixels "
                         "(base Qwen2.5-VL). 'norm': already 0-1000 (a student "
                         "fine-tuned on normalized labels).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_stream", dest="stream", action="store_false",
                    help="disable the live per-sample running-accuracy line "
                         "(printed to stderr).")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.hf_name, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    model = load_model(args.backend, args.model_path, args.adapter)
    results, by_type = run(ds, model, args.bbox_format, args.coord_space,
                           stream=args.stream)
    s = summarize(results)
    print(f"\nScreenSpot-V2 grounding accuracy: {s['accuracy']:.3f}  "
          f"({s['correct']}/{s['n']})")
    for t, rs in sorted(by_type.items()):
        ss = summarize(rs)
        print(f"  {t:10s} {ss['accuracy']:.3f}  ({ss['correct']}/{ss['n']})")


if __name__ == "__main__":
    main()
