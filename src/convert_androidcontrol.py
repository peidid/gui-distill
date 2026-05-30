"""Convert AndroidControl episodes into our `Step` schema (then -> ShareGPT).

Design: the TensorFlow/TFRecord reading is isolated in `iter_tfrecords()` so the
*conversion logic* (`record_to_steps`, `map_action`) is pure Python and unit-
tested on the Mac with a fake record — no TF, no network, no data download.

AndroidControl record fields (official schema):
  episode_id          : int
  goal                : str
  screenshots         : list[bytes]  (PNG)         len = N
  screenshot_widths   : list[int]                  len = N
  screenshot_heights  : list[int]                  len = N
  actions             : list[json str | dict]      len = N-1
  step_instructions   : list[str]                  len = N-1
Action dicts: {"action_type": "...", + type-specific keys}.
Coordinates in click/long_press are PIXELS of that screenshot -> normalized here.

Run (on the GPU box or a Mac with tensorflow installed + data downloaded):
    python src/convert_androidcontrol.py \
        --tfrecords "data/androidcontrol/android_control*" \
        --img_dir data/androidcontrol/imgs \
        --out data/androidcontrol/steps_train.jsonl \
        --max_episodes 500
"""

import argparse
import io
import json
import os

from action_space import (Action, CLICK, LONG_PRESS, TYPE, SCROLL, PRESS_BACK,
                           PRESS_HOME, OPEN_APP, WAIT, DONE)
from coords import norm_from_pixel
from schema import Step, save_steps

# AndroidControl action_type -> our action builder.
# Returns an Action, or None for action types we deliberately drop.
def map_action(a, img_w, img_h):
    if isinstance(a, (str, bytes)):
        a = json.loads(a)
    t = a.get("action_type")
    if t in ("click", "long_press"):
        x, y = norm_from_pixel(a["x"], a["y"], img_w, img_h)
        return Action(CLICK if t == "click" else LONG_PRESS, x=x, y=y)
    if t == "input_text":
        return Action(TYPE, text=a.get("text", ""))
    if t == "scroll":
        return Action(SCROLL, direction=a["direction"].lower())
    if t == "open_app":
        return Action(OPEN_APP, app=a.get("app_name") or a.get("app") or "")
    if t == "navigate_back":
        return Action(PRESS_BACK)
    if t == "navigate_home":
        return Action(PRESS_HOME)
    if t == "wait":
        return Action(WAIT)
    return None  # unknown/unsupported action_type -> caller skips the episode


def record_to_steps(rec, img_dir, append_done=True):
    """One AndroidControl record (dict) -> list[Step]. Writes screenshots to disk."""
    ep = rec["episode_id"]
    goal = rec["goal"]
    shots = rec["screenshots"]
    widths = rec["screenshot_widths"]
    heights = rec["screenshot_heights"]
    actions = rec["actions"]
    instrs = rec.get("step_instructions") or [None] * len(actions)

    os.makedirs(img_dir, exist_ok=True)
    steps, history = [], []

    for i, raw_action in enumerate(actions):
        w, h = widths[i], heights[i]
        act = map_action(raw_action, w, h)
        if act is None:
            return []  # drop the whole episode rather than train on a gap

        img_path = _write_png(shots[i], img_dir, f"ac{ep}_s{i}.png")
        steps.append(Step(
            episode_id=f"ac{ep}", step_idx=i, goal=goal, action=act,
            image_path=img_path, image_w=w, image_h=h,
            history=list(history),
            thought=(instrs[i] if i < len(instrs) else None),
        ))
        history.append(act)

    if append_done and len(shots) > len(actions):
        i = len(actions)  # the terminal screenshot
        img_path = _write_png(shots[i], img_dir, f"ac{ep}_s{i}.png")
        steps.append(Step(
            episode_id=f"ac{ep}", step_idx=i, goal=goal, action=Action(DONE),
            image_path=img_path, image_w=widths[i], image_h=heights[i],
            history=list(history), thought="The goal is complete.",
        ))
    return steps


def _write_png(png_bytes, img_dir, name):
    path = os.path.join(img_dir, name)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(png_bytes)
    return path


# --- TF-only boundary (not imported during tests) ------------------------------
def iter_tfrecords(glob_pattern, max_episodes=None):
    """Yield record dicts from AndroidControl TFRecords. Requires tensorflow."""
    import tensorflow as tf

    feature_desc = {
        "episode_id": tf.io.FixedLenFeature([], tf.int64),
        "goal": tf.io.FixedLenFeature([], tf.string),
        "screenshots": tf.io.VarLenFeature(tf.string),
        "screenshot_widths": tf.io.VarLenFeature(tf.int64),
        "screenshot_heights": tf.io.VarLenFeature(tf.int64),
        "actions": tf.io.VarLenFeature(tf.string),
        "step_instructions": tf.io.VarLenFeature(tf.string),
    }
    files = tf.io.gfile.glob(glob_pattern)
    ds = tf.data.TFRecordDataset(files, compression_type="GZIP")
    n = 0
    for raw in ds:
        ex = tf.io.parse_single_example(raw, feature_desc)
        sparse = lambda k: tf.sparse.to_dense(ex[k]).numpy()
        yield {
            "episode_id": int(ex["episode_id"].numpy()),
            "goal": ex["goal"].numpy().decode("utf-8"),
            "screenshots": list(sparse("screenshots")),
            "screenshot_widths": [int(x) for x in sparse("screenshot_widths")],
            "screenshot_heights": [int(x) for x in sparse("screenshot_heights")],
            "actions": [b.decode("utf-8") for b in sparse("actions")],
            "step_instructions": [b.decode("utf-8") for b in sparse("step_instructions")],
        }
        n += 1
        if max_episodes and n >= max_episodes:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecords", required=True, help="glob, e.g. 'data/androidcontrol/android_control*'")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--no_done", action="store_true", help="don't append terminal done() step")
    args = ap.parse_args()

    all_steps, kept, dropped = [], 0, 0
    for rec in iter_tfrecords(args.tfrecords, args.max_episodes):
        s = record_to_steps(rec, args.img_dir, append_done=not args.no_done)
        if s:
            all_steps.extend(s); kept += 1
        else:
            dropped += 1
    save_steps(all_steps, args.out)
    print(f"episodes kept={kept} dropped={dropped} | steps={len(all_steps)} -> {args.out}")


if __name__ == "__main__":
    main()
