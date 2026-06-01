"""Generate Track-B training labels with a teacher model (default UI-TARS-7B).

Teacher-forced: for each training Step, feed the SAME prompt the student sees
(screenshot + goal + ground-truth history), run the teacher, parse its output,
and emit a new Step whose action+thought are the TEACHER's. Output schema is
identical to steps_train.jsonl, so make_sharegpt / training are unchanged.

UI-TARS emits 0-1000 coords (verified on recon) — already canonical, so no
conversion. Steps whose teacher output is unparseable/unsupported are dropped
(never guessed) — that drop rate is itself a datapoint (teacher coverage).

    python src/run_teacher.py \
      --steps data/androidcontrol/steps_train.jsonl \
      --out   data/androidcontrol/steps_trackB.jsonl \
      --backend hf --teacher_model bytedance-research/UI-TARS-7B-DPO

then:
    python src/make_sharegpt.py data/androidcontrol/steps_trackB.jsonl \
           data/androidcontrol/sharegpt_trackB.json --abs
    # train a Track-B student by pointing the dataset at sharegpt_trackB.json
"""

import argparse
import re
import sys
from dataclasses import replace

from action_space import parse_step
from model import load_model
from prompt import SYSTEM, build_prompt
from schema import load_steps, save_steps
from uitars import parse_uitars

PARSERS = {"uitars": parse_uitars, "default": parse_step}

_THOUGHT = re.compile(r"Thought:\s*(.+?)\s*Action:", re.IGNORECASE | re.DOTALL)


def extract_thought(raw):
    m = _THOUGHT.search(raw)
    return m.group(1).strip() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True, help="input steps_*.jsonl (states to relabel)")
    ap.add_argument("--out", required=True, help="output steps_trackB.jsonl")
    ap.add_argument("--backend", default="hf", choices=["dummy", "mlx", "hf"])
    ap.add_argument("--teacher_model", default="bytedance-research/UI-TARS-7B-DPO")
    ap.add_argument("--parser", default="uitars", choices=list(PARSERS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    steps = load_steps(args.steps)
    if args.limit:
        steps = steps[:args.limit]
    model_path = None if args.backend == "dummy" else args.teacher_model
    model = load_model(args.backend, model_path)
    parse = PARSERS[args.parser]

    out, kept, dropped = [], 0, 0
    n = len(steps)
    for i, s in enumerate(steps, 1):
        raw = model.generate(s.image_path, SYSTEM, build_prompt(s))
        try:
            act = parse(raw)
        except Exception:
            act = None
        if act is None:
            dropped += 1
        else:
            # teacher's action + thought; gt_box was the human element box -> drop
            out.append(replace(s, action=act, thought=extract_thought(raw),
                               gt_box=None))
            kept += 1
        print(f"[{i}/{n}] kept={kept} dropped={dropped}"
              f"{' last=' + act.serialize() if act else ' (unparsed)'}",
              file=sys.stderr, flush=True)

    save_steps(out, args.out)
    print(f"teacher labels kept={kept} dropped={dropped} ({kept/n:.1%} coverage) "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
