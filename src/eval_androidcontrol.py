"""Offline step-accuracy eval on a steps.jsonl file (AndroidControl-test, or the
synthetic demo). Teacher-forced: each step's history is the ground-truth history,
so errors don't compound here — this measures per-step decision quality, the
single-step side of the gap.

Run (smoke test, no model):
    python src/eval_androidcontrol.py --steps data/synthetic/steps.jsonl --backend dummy
Run (real, on the GPU box with the trained student):
    python src/eval_androidcontrol.py --steps data/androidcontrol/steps_test.jsonl \
        --backend hf --model_path Qwen/Qwen2.5-VL-3B-Instruct --adapter out/qwen3b-trackA-lora
"""

import argparse
import json
from collections import defaultdict

from eval_core import step_correct, summarize
from model import load_model
from prompt import SYSTEM, build_prompt
from schema import load_steps


def run(steps, model, out_path=None, coord_space="pixel", stream=True):
    import sys
    n_total = len(steps) if hasattr(steps, "__len__") else None
    results, by_type, preds = [], defaultdict(list), []
    for i, s in enumerate(steps, 1):
        raw = model.generate(s.image_path, SYSTEM, build_prompt(s))
        ok = step_correct(raw, s.action, gt_box=s.gt_box,
                          coord_space=coord_space,
                          img_w=s.image_w, img_h=s.image_h)
        results.append(ok)
        by_type[s.action.type].append(ok)
        preds.append({"episode": s.episode_id, "step": s.step_idx,
                      "gt": s.action.serialize(), "raw": raw, "correct": ok})
        if stream:
            # running accuracy to stderr so it shows live even when stdout is
            # piped to `tee`, and never pollutes the final summary on stdout.
            run_acc = sum(results) / len(results)
            denom = f"/{n_total}" if n_total else ""
            mark = "OK " if ok else "miss"
            print(f"[{i}{denom}] acc={run_acc:.3f} ({sum(results)}/{len(results)})"
                  f"  {mark} {s.action.type}", file=sys.stderr, flush=True)
    if out_path:
        with open(out_path, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
    return results, by_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--backend", default="dummy", choices=["dummy", "mlx", "hf"])
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="write per-step predictions jsonl")
    ap.add_argument("--no_stream", dest="stream", action="store_false",
                    help="disable the live per-step running-accuracy line (stderr)")
    ap.add_argument("--coord_space", default="pixel", choices=["pixel", "norm"],
                    help="how to read the MODEL's click. 'pixel': absolute pixels "
                         "(base Qwen2.5-VL). 'norm': already 0-1000 (a student "
                         "fine-tuned on normalized labels). The dummy backend "
                         "works under either.")
    args = ap.parse_args()

    steps = load_steps(args.steps)
    if args.limit:
        steps = steps[:args.limit]
    model = load_model(args.backend, args.model_path, args.adapter)

    results, by_type = run(steps, model, args.out, args.coord_space, args.stream)
    s = summarize(results)
    print(f"\nAndroidControl-test step accuracy: {s['accuracy']:.3f}  "
          f"({s['correct']}/{s['n']})")
    print("per action type:")
    for t, rs in sorted(by_type.items()):
        ss = summarize(rs)
        print(f"  {t:12s} {ss['accuracy']:.3f}  ({ss['correct']}/{ss['n']})")


if __name__ == "__main__":
    main()
