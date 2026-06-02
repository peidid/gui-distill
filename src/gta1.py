"""Parse GTA1 (grounding specialist) output -> our canonical Action.

Recon facts (verified on GTA1-32B):
- COORDS ARE ABSOLUTE PIXELS (like base Qwen2.5-VL): e.g. click(962,1932) on a
  1080x2400 image, click(1574,569) on 1920x1080. Score/use with coord_space=
  'pixel' (normalize by image size). This is the OPPOSITE of UI-TARS (0-1000).
- Action syntax is OUR grammar -- click(x, y), long_press(x, y) -- but GTA1
  often appends a '# comment' (sometimes non-ASCII) after the call; we strip it
  by cutting at the ')' that closes the action call.
- GTA1 is a GROUNDING specialist: in practice it emits mostly click/long_press
  and rarely the discrete actions (type/scroll/open_app/done/wait). Those steps
  mismatch -- expected, part of the teacher action-coverage story.
"""

import re

from action_space import parse

_ACTION_LINE = re.compile(r"Action:\s*(.+)", re.IGNORECASE)


def parse_gta1(raw):
    """GTA1 raw completion -> our Action (coords are ABSOLUTE PIXELS)."""
    m = _ACTION_LINE.search(raw)
    if not m:
        raise ValueError("no Action: line found")
    act = m.group(1).strip().splitlines()[0].strip()
    # GTA1 sometimes appends '# comment' after the call -> cut at the first ')'
    # that closes the action call (fine for click/long_press numeric args).
    if ")" in act:
        act = act[: act.index(")") + 1]
    return parse(act)


if __name__ == "__main__":
    from action_space import Action, CLICK, LONG_PRESS
    cases = [
        ("Thought: ...\nAction: click(962, 1932)", Action(CLICK, x=962, y=1932)),
        ("Action: click(920, 91)", Action(CLICK, x=920, y=91)),
        ("Thought: ...\nAction: long_press(546, 1215)", Action(LONG_PRESS, x=546, y=1215)),
        ("Action: long_press(1000, 2006) # Clear the text field by pressing backspace多次",
         Action(LONG_PRESS, x=1000, y=2006)),
        ("Action: click(1574, 569)", Action(CLICK, x=1574, y=569)),
    ]
    for raw, exp in cases:
        got = parse_gta1(raw)
        assert got == exp, (raw, got, exp)
    for bad in ["Thought: just thinking, no action", "click without action prefix"]:
        try:
            parse_gta1(bad)
            assert False, f"expected ValueError for {bad!r}"
        except ValueError:
            pass
    print("gta1.py: all self-tests passed")
