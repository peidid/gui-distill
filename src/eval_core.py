"""Scoring logic for both evals. Pure Python (no model, no torch) so it is unit-
tested on the Mac. The model wrapper produces raw text; THIS decides correct/wrong.

Two metrics:
  - grounding (ScreenSpot-V2): model emits a click; correct iff the point lands
    inside the ground-truth element box (both in 0-1000 space).
  - step accuracy (AndroidControl-test): action TYPE must match, and the args
    must match (click within box-or-radius; text/scroll-dir/app string match).

A parse failure always counts as WRONG — a GUI agent that emits unparseable
actions is useless, so we never give it credit for "almost".
"""

import math

from action_space import parse_step, XY_ACTIONS, TYPE, SCROLL, OPEN_APP
from coords import Box, norm_from_pixel

# Default click tolerance for AndroidControl (no element box available):
# normalized L2 distance, 1000-grid. 100 ≈ 10% of screen width.
DEFAULT_CLICK_RADIUS = 100


def safe_parse(raw, parser=parse_step):
    """Model raw text -> Action, or None if unparseable.

    `parser` lets a different model's grammar be plugged in (e.g.
    uitars.parse_uitars for the Track-B teacher) without touching the scorers.
    """
    try:
        return parser(raw)
    except Exception:
        return None


def _norm_str(s):
    return (s or "").strip().lower()


def step_correct(raw, gt_action, gt_box=None, click_radius=DEFAULT_CLICK_RADIUS,
                 coord_space="norm", img_w=None, img_h=None, parser=parse_step):
    """AndroidControl-style step accuracy for one step.

    coord_space controls how the MODEL's click coords are read (the GT is always
    canonical 0-1000): 'norm' = already 0-1000 (a student trained on normalized
    labels, or UI-TARS); 'pixel' = absolute pixels (base Qwen2.5-VL) -> normalized
    here, which requires img_w/img_h. See coords.py for why the spaces must not be
    mixed. `parser` swaps the action grammar (e.g. uitars.parse_uitars).
    """
    pred = safe_parse(raw, parser)
    if pred is None or pred.type != gt_action.type:
        return False
    if gt_action.type in XY_ACTIONS:
        px, py = pred.x, pred.y
        if coord_space == "pixel":
            if img_w is None or img_h is None:
                raise ValueError("coord_space='pixel' requires img_w and img_h")
            px, py = norm_from_pixel(px, py, img_w, img_h)
        if gt_box:
            return Box(*gt_box).contains(px, py)
        d = math.hypot(px - gt_action.x, py - gt_action.y)
        return d <= click_radius
    if gt_action.type == TYPE:
        return _norm_str(pred.text) == _norm_str(gt_action.text)
    if gt_action.type == SCROLL:
        return pred.direction == gt_action.direction
    if gt_action.type == OPEN_APP:
        return _norm_str(pred.app) == _norm_str(gt_action.app)
    return True  # no-arg actions (press_back/home, wait, done): type match is enough


def grounding_correct(raw, gt_box, parser=parse_step):
    """ScreenSpot-V2: did the predicted click land in the GT element box (0-1000)?"""
    pred = safe_parse(raw, parser)
    if pred is None or pred.type not in XY_ACTIONS:
        return False
    return Box(*gt_box).contains(pred.x, pred.y)


def summarize(results):
    """results: list[bool] -> dict with n and accuracy."""
    n = len(results)
    correct = sum(1 for r in results if r)
    return {"n": n, "correct": correct, "accuracy": (correct / n if n else 0.0)}


if __name__ == "__main__":
    from action_space import Action
    # --- step accuracy ---
    # click within radius of GT point (no box)
    assert step_correct("Action: click(500, 500)", Action("click", x=520, y=515))
    assert not step_correct("Action: click(500, 500)", Action("click", x=900, y=900))
    # click judged by box when provided
    assert step_correct("Action: click(110, 110)", Action("click", x=0, y=0),
                        gt_box=(100, 100, 200, 200))
    # wrong action type
    assert not step_correct("Action: scroll(down)", Action("click", x=10, y=10))
    # text / scroll / app / noarg
    assert step_correct('Action: type("Hello")', Action("type", text="hello"))
    assert step_correct("Action: scroll(up)", Action("scroll", direction="up"))
    assert step_correct('Action: open_app("Chrome")', Action("open_app", app="chrome"))
    assert step_correct("Thought: done.\nAction: done()", Action("done"))
    # coord_space='pixel': model emits pixels, normalized before comparison
    # 540px of 1080 -> 500 (0-1000); inside box (450..550)
    assert step_correct("Action: click(540, 1200)", Action("click", x=0, y=0),
                        gt_box=(450, 450, 550, 550), coord_space="pixel",
                        img_w=1080, img_h=2400)
    # same pixel click read as 'norm' would be (540,1200) -> outside that box
    assert not step_correct("Action: click(540, 1200)", Action("click", x=0, y=0),
                            gt_box=(450, 450, 550, 550))
    # pixel mode without image size is a loud error, never a silent wrong
    try:
        step_correct("Action: click(5, 5)", Action("click", x=0, y=0),
                     coord_space="pixel")
        assert False, "expected ValueError"
    except ValueError:
        pass
    # unparseable -> wrong
    assert not step_correct("I will click the button", Action("click", x=10, y=10))
    # full thought+action accepted
    assert step_correct("Thought: tap it.\nAction: click(0,0)", Action("click", x=5, y=5))

    # --- grounding ---
    assert grounding_correct("Action: click(150, 150)", (100, 100, 200, 200))
    assert not grounding_correct("Action: click(300, 300)", (100, 100, 200, 200))
    assert not grounding_correct("Action: done()", (100, 100, 200, 200))

    # --- summary ---
    assert summarize([True, True, False, False])["accuracy"] == 0.5
    assert summarize([])["accuracy"] == 0.0
    print("eval_core.py: all self-tests passed")
