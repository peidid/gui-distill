"""The single source of truth for how a step is turned into model text.

CRITICAL: training and evaluation must build IDENTICAL prompts. If the student
is trained with one phrasing and evaluated with another, its scores collapse for
reasons that look like a model problem but are really a prompt mismatch. So both
sides import build_prompt() / build_target() from here — never inline a prompt.
"""

from action_space import Action

# Qwen2.5-VL expects the image referenced by a literal <image> token in the text.
IMAGE_TOKEN = "<image>"

SYSTEM = (
    "You are a mobile GUI agent. Given the phone screenshot, the user's goal, and "
    "the actions taken so far, output the single next action.\n"
    "Coordinates are integers in a 0-1000 grid (x=left->right, y=top->bottom).\n"
    "Reply with an optional 'Thought:' line then exactly one 'Action:' line.\n"
    "Allowed actions: click(x, y) | long_press(x, y) | type(\"text\") | "
    "scroll(up|down|left|right) | press_back() | press_home() | "
    "open_app(\"name\") | wait() | done()"
)


def build_prompt_parts(goal, history=()):
    """User-turn text from raw fields. Used by training, step eval, and grounding
    eval so all three share one exact phrasing."""
    hist = ", ".join(a.serialize() for a in history) if history else "(none)"
    return (
        f"{IMAGE_TOKEN}\n"
        f"Goal: {goal}\n"
        f"Actions so far: {hist}\n"
        f"What is the next action?"
    )


def build_prompt(step):
    """User-turn text for a Step (the screenshot is attached separately)."""
    return build_prompt_parts(step.goal, step.history)


def build_target(step):
    """Assistant-turn text (the training label / what we compare against)."""
    action_line = f"Action: {step.action.serialize()}"
    if step.thought:
        return f"Thought: {step.thought}\n{action_line}"
    return action_line


if __name__ == "__main__":
    from schema import Step
    s = Step(
        episode_id="ep1", step_idx=1, goal="Turn on wifi",
        action=Action("click", x=500, y=320),
        image_path="x.png", image_w=1080, image_h=2400,
        history=[Action("open_app", app="Settings")],
        thought="tap the wifi row",
    )
    print("--- PROMPT ---");  print(build_prompt(s))
    print("--- TARGET ---");  print(build_target(s))
    assert IMAGE_TOKEN in build_prompt(s)
    assert build_target(s).count("Action:") == 1
    print("\nprompt.py: self-test passed")
