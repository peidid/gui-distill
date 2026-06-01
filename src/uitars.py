"""Parse UI-TARS-7B output into our canonical `Action` schema (Track B teacher).

COORDINATE CONVENTION — verified empirically on a recon run: UI-TARS emits
**0-1000 normalized** coords (it is Qwen2-VL based). Its `click(start_box='(807,411)')`
matched the 0-1000 ground-truth `click(807,412)`. So UI-TARS coords are ALREADY in
our canonical space — score/use with coord_space='norm', do NOT divide by image
size. (This is the opposite of the pixel-native base/student Qwen2.5-VL.)

UI-TARS action syntax differs from ours; this parser maps the forms seen in recon
plus UI-TARS's documented action space:
    click(start_box='(x,y)')            -> click(x, y)
    long_press(start_box='(x,y)')       -> long_press(x, y)
    type(content='text')                -> type("text")
    scroll(direction='up') / scroll up()-> scroll(up)
    press_home() / press_back()         -> same
    finished() / finished(content=...)  -> done()
    wait()                              -> wait()
Unsupported UI-TARS actions (drag, hotkey, left_double, right_single, ...) raise
ValueError so they're scored as misses, never guessed at.
"""

import re

from action_space import (Action, CLICK, LONG_PRESS, TYPE, SCROLL, PRESS_BACK,
                           PRESS_HOME, WAIT, DONE)

_ACTION_LINE = re.compile(r"Action:\s*(.+)", re.IGNORECASE)


def _coords(s):
    nums = re.findall(r"-?\d+", s)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    raise ValueError(f"no coords in {s!r}")


def parse_uitars(raw):
    """UI-TARS raw completion -> our Action (coords in canonical 0-1000)."""
    m = _ACTION_LINE.search(raw)
    if not m:
        raise ValueError("no Action: line found")
    act = m.group(1).strip().splitlines()[0].strip()
    low = act.lower()

    if low.startswith(("click", "left_single", "left_click")):
        x, y = _coords(act)
        return Action(CLICK, x=x, y=y)
    if low.startswith(("long_press", "longpress")):
        x, y = _coords(act)
        return Action(LONG_PRESS, x=x, y=y)
    if low.startswith("type"):
        q = re.search(r"content\s*=\s*['\"](.*)['\"]\s*\)?\s*$", act, re.DOTALL)
        text = q.group(1) if q else re.sub(r"^type\s*\(|\)$", "", act).strip(" '\"")
        return Action(TYPE, text=text)
    if low.startswith("scroll"):
        d = re.search(r"(up|down|left|right)", low)
        if not d:
            raise ValueError(f"scroll direction not found: {act!r}")
        return Action(SCROLL, direction=d.group(1))
    if low.startswith(("press_home", "home(")):
        return Action(PRESS_HOME)
    if low.startswith(("press_back", "back(")):
        return Action(PRESS_BACK)
    if low.startswith(("finished", "done", "complete")):
        return Action(DONE)
    if low.startswith("wait"):
        return Action(WAIT)
    raise ValueError(f"unsupported UI-TARS action: {act!r}")


if __name__ == "__main__":
    # Self-tests use the ACTUAL strings captured from UI-TARS-7B-DPO on recon.
    cases = [
        ("Thought: ...press the home button\nAction: press_home()", Action(PRESS_HOME)),
        ("Thought: ...scroll up\nAction: scroll up()", Action(SCROLL, direction="up")),
        ("Thought: three dot icon\nAction: click(start_box='(950,87)')", Action(CLICK, x=950, y=87)),
        ("Thought: edit name\nAction: click(start_box='(499,505)')", Action(CLICK, x=499, y=505)),
        ("Thought: type it\nAction: type(content='Audio 1')", Action(TYPE, text="Audio 1")),
        ("Thought: rename\nAction: click(start_box='(807,411)')", Action(CLICK, x=807, y=411)),
        ("Action: click(start_box='(969,174)')", Action(CLICK, x=969, y=174)),
        # documented UI-TARS variants not seen in recon but supported:
        ("Action: scroll(direction='down')", Action(SCROLL, direction="down")),
        ("Action: finished()", Action(DONE)),
        ("Action: finished(content='all set')", Action(DONE)),
        ("Action: wait()", Action(WAIT)),
        ("Action: long_press(start_box='(120, 640)')", Action(LONG_PRESS, x=120, y=640)),
    ]
    for raw, exp in cases:
        got = parse_uitars(raw)
        assert got == exp, (raw, got, exp)

    # unsupported / unparseable -> loud failure (counted as miss downstream)
    for bad in ["Action: drag(start_box='(1,2)', end_box='(3,4)')",
                "Action: hotkey(key='ctrl c')",
                "I will click the button"]:
        try:
            parse_uitars(bad)
            assert False, f"expected ValueError for {bad!r}"
        except ValueError:
            pass

    print("uitars.py: all self-tests passed")
