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

import json
import sys

from prompt import SYSTEM, build_prompt, build_target, IMAGE_TOKEN
from schema import load_steps


def step_to_row(step):
    user = build_prompt(step)
    assert user.count(IMAGE_TOKEN) == 1, "exactly one <image> token required"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": build_target(step)},
        ],
        "images": [step.image_path],
    }


def convert(steps_path, out_path):
    steps = load_steps(steps_path)
    rows = [step_to_row(s) for s in steps]
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    return len(rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python src/make_sharegpt.py <steps.jsonl> <out.json>")
        sys.exit(1)
    n = convert(sys.argv[1], sys.argv[2])
    print(f"wrote {n} ShareGPT rows -> {sys.argv[2]}")
