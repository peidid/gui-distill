# Mobile GUI Agent Distillation — learning run

Goal: distill a small (3B) mobile GUI agent from a stronger teacher and **feel the
grounding-vs-multi-step gap firsthand**. This repo is the hands-on build behind a
planned 2026 ML-venue paper on *black-box, hard-label distillation of GUI agents*.

Black-box hard-label distillation here = run a strong teacher to produce
`Thought: … Action: …` trajectories, then plain-text SFT a small student on them.
No logits, no special loss. Everything below is plumbing around that one idea.

## Compute split (Mac + ~$20 budget)

| Where | Cost | What runs there |
|-------|------|-----------------|
| **MacBook Air M4 (this box)** | $0 | Stage 0 spec, data conversion, ScreenSpot-V2 + AndroidControl-test eval, orchestration |
| **Rented Linux + NVIDIA (one burst)** | ~$5–15 | Stage 3 LoRA SFT of Qwen2.5-VL-3B; optional 7B teacher rollouts |
| **AndroidWorld (live emulator)** | deferred | needs x86 + KVM; skip for the learning run, use AndroidControl-test as the proxy |

## Stack
- **Student:** Qwen2.5-VL-3B-Instruct (realistic on-device size, native 0–1000 grounding)
- **Teacher (learning run):** GUI-Owl-7B or UI-TARS-7B (cheap open rollouts). Closed API teacher is for the *paper*, not this run.
- **Train:** LLaMA-Factory, LoRA, single GPU
- **Data:** AndroidControl (human demos: screenshot + instruction + action)
- **Eval:** ScreenSpot-V2 (static grounding) + AndroidControl test split (offline step accuracy). AndroidWorld deferred.

## The one rule that saves a week
Normalize **all** coordinates to the **0–1000 integer space** (Qwen's native convention),
everywhere: teacher output, training labels, eval parsing. AndroidControl pixel coords,
Qwen grounding outputs, and emulator screen coords are *different spaces*. Convert once,
in `src/coords.py`, with tests. See the Qwen2.5-VL bbox-scaling gotcha noted there.

## Stages
- **Stage 0 — DONE-ABLE NOW:** action vocab + coord normalization + tests → `src/action_space.py`, `src/coords.py`
- **Stage 1 — Mac:** eval harness (ScreenSpot-V2, AndroidControl-test) on the *base* student first → baselines
- **Stage 2 — Mac:** Track A = AndroidControl→ShareGPT (behavior cloning, no teacher). Track B = teacher rollouts.
- **Stage 3 — rented GPU:** LoRA SFT the student
- **Stage 4 — Mac:** eval student; put base / student / teacher in one table → see the gap
- **Stage 5 — optional:** pure vs. rejection-sampled traces ablation (teacher-error inheritance)

## Layout
```
src/action_space.py   # fixed action vocabulary, parse <-> serialize
src/coords.py         # 0–1000 normalization, pixel <-> norm conversions
tests/                # self-contained, run with `python3 tests/<file>.py` (no pytest needed)
```
