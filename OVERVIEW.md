# Project Overview — Distilling a Mobile GUI Agent

A plain-language map of *what* this project is, *why* it exists, and *how the
pieces fit together*. Read this first; then follow `HANDS_ON.md` to build it.

---

## 1. The one-sentence version

We take a **strong but expensive "teacher"** model that can operate a phone by
looking at screenshots, have it produce step-by-step demonstrations, and use
those to **train a small "student"** model that can run on-device — then we
**measure exactly where the student falls short of the teacher.**

That last part — *where and why the small model fails* — is the research.

---

## 2. What is a "GUI agent"?

A GUI (Graphical User Interface) agent is a model that:
1. **Sees** a screenshot of a phone screen.
2. **Reads** a goal in plain English ("Turn on Wi-Fi").
3. **Outputs an action** — e.g. `click(500, 320)`, `type("hello")`, `scroll(down)`.
4. Repeats, step by step, until the task is done.

So it's a loop: *screenshot + goal → action → new screenshot → action → …*
Each action is a tap/type/scroll at specific screen coordinates.

Two skills matter, and we test them separately:
- **Grounding** (one-shot perception): "given this screen, where is the search
  bar?" → output a coordinate. Tested by **ScreenSpot-V2**.
- **Multi-step control** (a whole task): "book a hotel" → a *sequence* of correct
  actions where any wrong step derails the rest. Tested by **AndroidControl**.

A model can be good at grounding but bad at multi-step, because errors **compound**
over a long task. Holding those two apart is central to this project.

---

## 3. What is "distillation," and why "black-box / hard-label"?

**Distillation** = training a small model to imitate a big one.

- The textbook version ("soft-label") copies the big model's internal probability
  distribution over every token. It needs access to the model's *logits* (its raw
  internal numbers).
- But the strongest GUI teachers are **closed APIs** (GPT, Claude, Gemini). You
  send a screenshot, you get back text. **No logits.** This is the *black-box*
  setting. All you can use is the final text the teacher emits — its **hard
  labels**.

So **black-box hard-label distillation** is just:
> Run the teacher to get `Thought: … Action: …` text, then do ordinary
> supervised fine-tuning (SFT) of the student on that text.

No special algorithm. The research question is not "what clever loss?" but
**"what capability do you LOSE when you're restricted to this simple, realistic
setup — and does the loss concentrate in grounding, in reasoning, or in
multi-step robustness?"**

---

## 4. The research framing (for the paper)

- **Venue:** an ML venue (NeurIPS / ICML / ICLR / ACL / COLM), *not* CHI — there's
  no human-study/interaction contribution here.
- **The crowded parts (avoid claiming as novel):** that grounding vs. reasoning
  transfer differently (already published); that a static↔interactive gap exists
  (already visible in published tables); that rejection sampling helps (well-known).
- **The open wedge (the actual contribution):** the **black-box, API-only**
  constraint. A rigorous study of *what hard-label distillation from a genuinely
  closed teacher costs for GUI agents specifically, and where that cost
  concentrates.*
- **Premise to protect:** the teacher must be genuinely closed (a real API). If
  you distill from an open-weight teacher, a reviewer will say "you had logits,
  this isn't black-box." For *learning the pipeline* we use an open teacher
  (cheap, fast); for the *paper* we swap in a closed API.

**This repo is the learning run** — build the whole thing end to end with cheap
parts so you understand every moving piece *before* you spend real compute and
make the premise rigorous.

---

## 5. The models

| Role | Model | Why |
|---|---|---|
| **Student** | Qwen2.5-VL-3B-Instruct | Realistic on-device size; well supported by training tools. **Coordinate gotcha:** unlike Qwen2-VL/Qwen-VL, Qwen2.5-VL emits *absolute pixel* coordinates (of the smart-resized image), **not** 0–1000. We normalize everything to the canonical 0–1000 grid in code — see the coordinate design rule in §7. |
| **Track A teacher** | AndroidControl human demos | Correct human actions already in the dataset — no teacher inference. The (≈error-free) reference. |
| **Track B teacher** | UI-TARS-7B (open) | A capable mid-size GUI agent; emits `Thought:/Action:` natively (parser: `src/uitars.py`; emits **0–1000** coords). ~14 GB → fits one 24 GB GPU. |
| **Track C teacher** | **UI-Venus-72B** (primary), **GTA1-32B** (strong alt) | A much stronger GUI agent → tests whether a better teacher yields a better student, or the 3B student is capacity-bound. Needs ~80 GB (A100/H100); each needs its own recon + parser. |
| **Paper teacher (future)** | a closed API (GPT/Claude/Gemini) | Makes the black-box premise rigorous — open weights ≠ black-box. A later track once the open-teacher study is solid. |

---

## 6. The data

- **AndroidControl** — 15k+ human demonstrations on 800+ real Android apps.
  Each episode = a goal + a sequence of (screenshot, action) pairs. This is both
  our **training source** and our **multi-step test set**.
- **ScreenSpot-V2** — ~1.2k (instruction, screenshot, target box) samples. Pure
  **grounding** test. No emulator needed.
- **AndroidWorld** (deferred) — a *live emulator* with 116 interactive tasks. The
  gold standard for multi-step, but needs x86+KVM hardware. We use AndroidControl's
  test split as a cheaper proxy for now.

---

## 7. How the code is organized (and why it's split this way)

The guiding principle: **isolate the parts that need heavy infrastructure
(GPU, network, TensorFlow) from the pure logic, so the logic can be unit-tested
on a laptop.** That's why nearly everything has a `python file.py` self-test.

```
src/
  coords.py            ← 0–1000 coordinate math. ONE place all coords convert.
  action_space.py      ← the fixed action vocabulary + text<->object parsing.
  schema.py            ← Step: the common record every dataset funnels into.
  prompt.py            ← the ONE prompt format shared by training & eval.
  convert_androidcontrol.py ← raw dataset -> Steps (TF read isolated from logic).
  make_sharegpt.py     ← Steps -> training rows (LLaMA-Factory format).
  eval_core.py         ← scoring rules (pure, no model).
  model.py             ← dummy / mlx / hf inference backends.
  eval_screenspot.py   ← grounding eval driver.
  eval_androidcontrol.py ← multi-step (offline) eval driver.
configs/               ← LLaMA-Factory training config + dataset registration.
scripts/               ← synthetic demo generator + the full GPU-burst recipe.
```

### The data flow, start to finish
```
                 ┌─────────────── on the Mac (free) ───────────────┐
raw AndroidControl ─▶ convert_androidcontrol ─▶ Step (jsonl) ─▶ make_sharegpt ─▶ training rows
                                                   │
                                                   └─▶ (used directly by the eval drivers too)

                 ┌────────── on the rented GPU (~$2) ──────────┐
training rows ─▶ LLaMA-Factory LoRA SFT ─▶ student adapter
                                              │
base & student & teacher ─▶ eval drivers ─▶ ScreenSpot + AndroidControl numbers ─▶ THE TABLE
```

### Two design rules that prevent the most common failures
1. **All coordinates live in a single 0–1000 integer space.** Pixel coords from
   the dataset, the model's outputs, and the emulator are three *different* spaces;
   mixing them trains a model that clicks the wrong place while the loss looks
   fine. Convert once, in `coords.py`, with tests.
   *Two real instances of this bit us and are now handled in `eval_screenspot.py`:*
   (a) the `HongxinLi/ScreenSpot_v2` ground-truth boxes are stored as **normalized
   `[0,1]` fractions (xyxy)**, not pixels — `bbox_to_norm` auto-detects and scales
   them; (b) the **base Qwen2.5-VL emits absolute pixels** while a **student
   fine-tuned on our 0–1000 labels emits 0–1000** — so the eval takes a
   `--coord_space {pixel,norm}` flag (`pixel` for the base model, `norm` for the
   student). Get this wrong and accuracy reads a flat 0.000 even when the model is
   clicking the right place.
2. **Training and evaluation build the *identical* prompt** (`prompt.py`). If they
   differ, the student is judged on a format it never saw and scores look broken
   for a non-model reason.

---

## 8. The experiment & the expected result

**Design.** Hold the **student** (Qwen2.5-VL-3B) and the **recipe** (same LoRA
hyperparameters, same training states, same eval) fixed, and vary **only the
teacher** — i.e. who produces the action labels. Any difference is then
attributable to the teacher. This gives a clean *teacher* sweep:

| Track | Teacher (label source) | What it tests |
|---|---|---|
| **A** | AndroidControl human demos | best case — correct (≈error-free) labels |
| **B** | UI-TARS-7B (open) | distillation from a *capable* model teacher |
| **C** | UI-Venus-72B (primary) / GTA1-32B (alt) | does a *stronger* teacher yield a better student, or is the 3B student capacity-bound? |

Fill this table — every cell scored through the *identical* pipeline:

```
                          ScreenSpot-V2   AndroidControl-test
                          (grounding)     (multi-step)
base Qwen2.5-VL-3B           0.603           0.114     (measured)
student (A: human)           0.280           0.552     (measured)
student (B: UI-TARS-7B)      __              __
student (C: 72B / 32B)       __              __
teacher B (UI-TARS-7B)       __              __        ← B's ceiling
teacher C (72B / 32B)        __              __        ← C's ceiling
```

What the sweep measures:
- **Static vs interactive gap.** Does the student close the *grounding* gap more
  than the *multi-step* gap (or vice-versa)? Per-action-type breakdowns turn "a
  gap exists" into "here is the mechanism" — the publishable part. *(Track A
  already showed the opposite of the naive guess: multi-step ↑×4.8 in-distribution
  while out-of-distribution grounding ↓ by half — see `results/Summary.txt`.)*
- **Teacher-error inheritance (A vs B/C).** Human labels are ≈correct; model
  teachers make mistakes the student can inherit. A→B→C is a gradient of teacher
  error.
- **Teacher-capability transfer (B vs C).** Plot *student accuracy vs measured
  teacher accuracy*. If the 72B teacher barely beats the 7B-taught student, the
  bottleneck is the student / black-box channel, not the teacher — a real finding.

Controls that keep it clean: vary only the teacher; **relabel the *same* training
states** for B and C (only the labels differ); and **measure each teacher's own
accuracy** (the teacher rows) so student quality is related to *measured* teacher
skill, not just parameter count. Each new teacher needs its own output recon +
parser (different action syntax / coordinate convention — see `src/uitars.py` for
UI-TARS, which emits 0–1000).

**Premise to protect:** A/B/C all use *open* teachers — they are the **learning
run**. The paper's headline teacher must be a genuinely **closed API** (open
weights ≠ black-box); that becomes a later track once the open-teacher mechanism
study is solid.

---

## 9. Glossary

- **VLM** — Vision-Language Model: takes images + text, outputs text.
- **SFT** — Supervised Fine-Tuning: train on (input, desired-output) pairs.
- **LoRA** — a cheap fine-tuning method that trains small adapter weights instead
  of the whole model; fits a 3B model on one consumer GPU.
- **ShareGPT format** — a JSON chat format (`messages` + `images`) that
  LLaMA-Factory reads for training.
- **Grounding** — mapping an instruction to a screen coordinate.
- **Rollout / trajectory** — one full run of an agent on a task (the sequence of
  thoughts + actions).
- **Rejection sampling** — keep only the teacher trajectories that actually
  succeeded; throw away the failures before training.
- **Teacher forcing** — during offline eval, feed the *ground-truth* history at
  each step (so errors don't compound), to measure per-step quality in isolation.
```
