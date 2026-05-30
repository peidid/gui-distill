"""Test the AndroidControl conversion LOGIC with a fake record (no TF, no download).
Run: python tests/test_convert.py
"""
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PIL import Image

from action_space import Action, CLICK, OPEN_APP, DONE
from convert_androidcontrol import map_action, record_to_steps
from coords import norm_from_pixel


def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 200, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_map_action():
    W, H = 1000, 2000
    # click pixels -> normalized 0-1000: 500/1000*1000=500, 1000/2000*1000=500
    a = map_action({"action_type": "click", "x": 500, "y": 1000}, W, H)
    assert a == Action(CLICK, x=500, y=500), a
    # the dataset's other types map cleanly
    assert map_action({"action_type": "input_text", "text": "hi"}, W, H) == Action("type", text="hi")
    assert map_action({"action_type": "navigate_back"}, W, H) == Action("press_back")
    assert map_action({"action_type": "navigate_home"}, W, H) == Action("press_home")
    assert map_action({"action_type": "open_app", "app_name": "Chrome"}, W, H) == Action(OPEN_APP, app="Chrome")
    assert map_action({"action_type": "scroll", "direction": "Down"}, W, H) == Action("scroll", direction="down")
    # JSON string input also accepted
    assert map_action(json.dumps({"action_type": "wait"}), W, H) == Action("wait")
    # unknown -> None
    assert map_action({"action_type": "teleport"}, W, H) is None


def test_record_to_steps():
    W, H = 1080, 2400
    rec = {
        "episode_id": 42,
        "goal": "Open Chrome and search",
        "screenshots": [_png(W, H), _png(W, H), _png(W, H)],   # N=3
        "screenshot_widths": [W, W, W],
        "screenshot_heights": [H, H, H],
        "actions": [                                            # N-1 = 2
            {"action_type": "open_app", "app_name": "Chrome"},
            {"action_type": "click", "x": 540, "y": 1200},
        ],
        "step_instructions": ["Launch Chrome", "Tap the search bar"],
    }
    with tempfile.TemporaryDirectory() as d:
        steps = record_to_steps(rec, d, append_done=True)
    # 2 actions + 1 appended done()
    assert len(steps) == 3, len(steps)
    assert steps[0].action == Action(OPEN_APP, app="Chrome")
    assert steps[0].history == []
    assert steps[1].action == Action(CLICK, *norm_from_pixel(540, 1200, W, H))
    assert steps[1].history == [Action(OPEN_APP, app="Chrome")]
    assert steps[1].thought == "Tap the search bar"
    assert steps[2].action == Action(DONE)
    assert len(steps[2].history) == 2  # saw both prior actions


def test_drops_episode_with_unsupported_action():
    W, H = 800, 1600
    rec = {
        "episode_id": 7, "goal": "x",
        "screenshots": [_png(W, H), _png(W, H)],
        "screenshot_widths": [W, W], "screenshot_heights": [H, H],
        "actions": [{"action_type": "teleport"}],
        "step_instructions": ["?"],
    }
    with tempfile.TemporaryDirectory() as d:
        assert record_to_steps(rec, d) == []


if __name__ == "__main__":
    test_map_action()
    test_record_to_steps()
    test_drops_episode_with_unsupported_action()
    print("test_convert.py: all tests passed")
