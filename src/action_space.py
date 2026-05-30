"""Fixed action vocabulary for the mobile GUI agent + parse <-> serialize.

Pin this BEFORE generating any data. The teacher emits actions as text, the
student is trained on that text, and the evaluator parses that text — all three
must agree on exactly one grammar. Coordinates are always canonical 0-1000 ints
(see coords.py).

Canonical serialized forms (what the model reads and writes):
    click(500, 320)
    type("hello world")
    scroll(down)
    press_back()
    open_app("Settings")
    wait()
    done()

A full step the model produces looks like:
    Thought: I need to open settings to change the wallpaper.
    Action: open_app("Settings")

The parser is deliberately tolerant of teacher quirks (keyword args, single
quotes, extra spaces) but `serialize()` only ever emits the canonical form, so
training labels are uniform regardless of which teacher produced them.
"""

import re
from dataclasses import dataclass, field

# --- vocabulary ----------------------------------------------------------------
# Chosen to map LOSSLESSLY onto AndroidControl's action types:
#   click/long_press -> x,y ;  input_text -> type ;  navigate_back -> press_back ;
#   navigate_home -> press_home ;  scroll/open_app/wait map directly.
# `done` is ours: appended at episode end to teach the agent to terminate.
CLICK = "click"
LONG_PRESS = "long_press"
TYPE = "type"
SCROLL = "scroll"
PRESS_BACK = "press_back"
PRESS_HOME = "press_home"
OPEN_APP = "open_app"
WAIT = "wait"
DONE = "done"

ACTION_TYPES = {CLICK, LONG_PRESS, TYPE, SCROLL, PRESS_BACK, PRESS_HOME,
                OPEN_APP, WAIT, DONE}
XY_ACTIONS = {CLICK, LONG_PRESS}          # actions carrying coordinates
NOARG_ACTIONS = {PRESS_BACK, PRESS_HOME, WAIT, DONE}
SCROLL_DIRS = {"up", "down", "left", "right"}


@dataclass(frozen=True)
class Action:
    type: str
    x: int = None          # click
    y: int = None          # click
    text: str = None       # type
    direction: str = None  # scroll
    app: str = None        # open_app

    def __post_init__(self):
        if self.type not in ACTION_TYPES:
            raise ValueError(f"unknown action type: {self.type!r}")
        if self.type in XY_ACTIONS and (self.x is None or self.y is None):
            raise ValueError(f"{self.type} requires x and y")
        if self.type == TYPE and self.text is None:
            raise ValueError("type requires text")
        if self.type == SCROLL and self.direction not in SCROLL_DIRS:
            raise ValueError(f"scroll direction must be one of {SCROLL_DIRS}")
        if self.type == OPEN_APP and not self.app:
            raise ValueError("open_app requires app name")

    def serialize(self):
        if self.type in XY_ACTIONS:
            return f"{self.type}({self.x}, {self.y})"
        if self.type == TYPE:
            return f'type({_q(self.text)})'
        if self.type == SCROLL:
            return f"scroll({self.direction})"
        if self.type == OPEN_APP:
            return f"open_app({_q(self.app)})"
        return f"{self.type}()"  # press_back, press_home, wait, done


def _q(s):
    """Serialize a string arg with escaped double quotes."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _unq(s):
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1]
    return s.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")


_CALL_RE = re.compile(r"^\s*([a-z_]+)\s*\((.*)\)\s*$", re.DOTALL)


def parse(text):
    """Parse one serialized action into an Action.

    Accepts the canonical form plus common teacher variants:
    `click(x=500, y=320)`, single quotes, stray whitespace. Raises ValueError
    on anything it can't map to the fixed vocabulary — fail loud, never guess.
    """
    m = _CALL_RE.match(text)
    if not m:
        raise ValueError(f"not an action call: {text!r}")
    name, args = m.group(1), m.group(2).strip()

    if name in XY_ACTIONS:
        x, y = _parse_xy(args)
        return Action(name, x=x, y=y)
    if name == TYPE:
        return Action(TYPE, text=_unq(args))
    if name == SCROLL:
        return Action(SCROLL, direction=_unq(args).lower())
    if name == OPEN_APP:
        return Action(OPEN_APP, app=_unq(args))
    if name in NOARG_ACTIONS:
        return Action(name)
    raise ValueError(f"unknown action: {name!r}")


def _parse_xy(args):
    # supports "500, 320" and "x=500, y=320" (any order)
    kv = dict(re.findall(r"([xy])\s*=\s*(-?\d+)", args))
    if "x" in kv and "y" in kv:
        return int(kv["x"]), int(kv["y"])
    nums = re.findall(r"-?\d+", args)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    raise ValueError(f"click needs two coords, got {args!r}")


# Step-level helpers: the model's full output is "Thought: ...\nAction: ...".
_ACTION_LINE_RE = re.compile(r"Action:\s*(.+)\s*$", re.IGNORECASE | re.DOTALL)


def parse_step(raw):
    """Extract the Action from a full 'Thought: ... Action: ...' completion."""
    m = _ACTION_LINE_RE.search(raw.strip())
    if not m:
        raise ValueError("no Action: line found")
    # take only the first line of the action (guard against trailing text)
    action_text = m.group(1).strip().splitlines()[0]
    return parse(action_text)


def format_step(thought, action):
    return f"Thought: {thought}\nAction: {action.serialize()}"


if __name__ == "__main__":
    # Self-tests — run: python3 src/action_space.py
    cases = [
        Action(CLICK, x=500, y=320),
        Action(TYPE, text='say "hi"'),
        Action(SCROLL, direction="down"),
        Action(PRESS_BACK),
        Action(OPEN_APP, app="Settings"),
        Action(WAIT),
        Action(DONE),
    ]
    # round-trip: serialize -> parse -> identical
    for a in cases:
        assert parse(a.serialize()) == a, (a, a.serialize(), parse(a.serialize()))

    # tolerant parsing of teacher variants
    assert parse("click(x=10, y=20)") == Action(CLICK, x=10, y=20)
    assert parse("click( 10 ,20 )") == Action(CLICK, x=10, y=20)
    assert parse("type('hello')") == Action(TYPE, text="hello")
    assert parse("scroll(Down)") == Action(SCROLL, direction="down")
    assert parse("open_app('Google Maps')") == Action(OPEN_APP, app="Google Maps")

    # full step parse
    a = parse_step("Thought: open the app first.\nAction: open_app(\"Clock\")")
    assert a == Action(OPEN_APP, app="Clock"), a
    assert parse_step("Action: done()") == Action(DONE)

    # invalid inputs fail loud
    for bad in ["frobnicate(1)", "click(5)", "scroll(sideways)", "not a call"]:
        try:
            parse(bad)
            assert False, f"expected ValueError for {bad!r}"
        except ValueError:
            pass

    # format_step shape
    s = format_step("tap submit", Action(CLICK, x=100, y=900))
    assert s == "Thought: tap submit\nAction: click(100, 900)", s

    print("action_space.py: all self-tests passed")
