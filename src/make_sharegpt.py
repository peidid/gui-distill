"""Convert a steps.jsonl file into a ShareGPT JSON file for LLaMA-Factory.

ShareGPT row shape LLaMA-Factory expects for a VLM:
    {
      "messages": [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": "<image>\nGoal: ...\n..."},
        {"role": "assistant", "content": "Thought: ...\nAction: click(500, 320)"}
      ],
      "images": ["data/.../screen.png"]
    }

The number of <image> tokens in the text must equal len(images) — here exactly 1.

Usage:
    python src/make_sharegpt.py data/synthetic/steps.jsonl data/synthetic/sharegpt.json
"""

import argparse
import json
import os

from prompt import SYSTEM, build_prompt, build_target, IMAGE_TOKEN
from schema import load_steps


def step_to_row(step, abs_paths=False):
    user = build_prompt(step)
    assert user.count(IMAGE_TOKEN) == 1, "exactly one <image> token required"
    img = os.path.abspath(step.image_path) if abs_paths else step.image_path
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": build_target(step)},
        ],
        "images": [img],
    }


def convert(steps_path, out_path, abs_paths=False):
    steps = load_steps(steps_path)
    rows = [step_to_row(s, abs_paths) for s in steps]
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps")
    ap.add_argument("out")
    ap.add_argument("--abs", action="store_true", dest="abs_paths",
                    help="write ABSOLUTE image paths (use for LLaMA-Factory training "
                         "to avoid path-resolution issues)")
    args = ap.parse_args()
    n = convert(args.steps, args.out, args.abs_paths)
    kind = "absolute" if args.abs_paths else "relative"
    print(f"wrote {n} ShareGPT rows ({kind} image paths) -> {args.out}")
