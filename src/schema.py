"""The intermediate representation that everything funnels through.

Raw datasets (AndroidControl) and synthetic data both get parsed into a list of
`Step`s. The ShareGPT converter and the evaluators only ever see `Step`s, so we
write the messy dataset-specific parsing exactly once and everything downstream
stays clean.

Invariant: a Step's action is ALREADY in canonical 0-1000 coords (see coords.py).
Normalization happens at the moment of parsing the raw dataset, never later.
"""

import json
from dataclasses import dataclass, field, asdict

from action_space import Action, parse as parse_action


@dataclass
class Step:
    episode_id: str        # which task episode this step belongs to
    step_idx: int          # position within the episode (0-based)
    goal: str              # high-level task instruction ("Turn on wifi")
    action: Action         # the TARGET action, in canonical 0-1000 coords
    image_path: str        # screenshot path, relative to repo root
    image_w: int           # original screenshot width  (px) — for eval back-conversion
    image_h: int           # original screenshot height (px)
    history: list = field(default_factory=list)  # prior actions this episode, as Action
    thought: str = None    # optional reasoning text (teacher trace, or low-level instruction)
    gt_box: tuple = None    # optional (x1,y1,x2,y2) in 0-1000 of the target element, for click eval

    def to_json(self):
        return {
            "episode_id": self.episode_id,
            "step_idx": self.step_idx,
            "goal": self.goal,
            "action": self.action.serialize(),
            "image_path": self.image_path,
            "image_w": self.image_w,
            "image_h": self.image_h,
            "history": [a.serialize() for a in self.history],
            "thought": self.thought,
            "gt_box": list(self.gt_box) if self.gt_box else None,
        }

    @classmethod
    def from_json(cls, d):
        return cls(
            episode_id=d["episode_id"],
            step_idx=d["step_idx"],
            goal=d["goal"],
            action=parse_action(d["action"]),
            image_path=d["image_path"],
            image_w=d["image_w"],
            image_h=d["image_h"],
            history=[parse_action(s) for s in d.get("history", [])],
            thought=d.get("thought"),
            gt_box=tuple(d["gt_box"]) if d.get("gt_box") else None,
        )


def save_steps(steps, path):
    with open(path, "w") as f:
        for s in steps:
            f.write(json.dumps(s.to_json()) + "\n")


def load_steps(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Step.from_json(json.loads(line)))
    return out


if __name__ == "__main__":
    # round-trip self-test — run: python3 src/schema.py
    s = Step(
        episode_id="ep1", step_idx=2, goal="Turn on wifi",
        action=Action("click", x=500, y=320),
        image_path="data/x.png", image_w=1080, image_h=2400,
        history=[Action("open_app", app="Settings")],
        thought="tap the wifi row", gt_box=(450, 300, 550, 340),
    )
    d = s.to_json()
    s2 = Step.from_json(json.loads(json.dumps(d)))
    assert s2.action == s.action and s2.history == s.history and s2.gt_box == s.gt_box
    print("schema.py: round-trip self-test passed")
